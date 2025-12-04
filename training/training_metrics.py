import torch
import torch.nn.functional as F
import numpy as np
import os
import time
from math import ceil, floor

from pytorch3d.structures import Pointclouds
from pytorch3d.ops import iterative_closest_point

from lightning.pytorch.loggers import WandbLogger

from hydra.utils import instantiate
from dataclasses import dataclass
from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.vggt.utils.geometry import closed_form_inverse_se3
#from vggt.training.train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from aligned_vggt.utils.geometry import unproject_depth_map_to_point_map
from aligned_vggt.utils.visualization import viser_wrapper

from aligned_vggt.utils.data import alignAndConvertOutputs, generate_chunks, chunk_batch, moveDictListItemToCPU, pose_encoding_to_extri, normalize_camera_extrinsics_and_points_batch

@dataclass(eq=False)
class Metrics(torch.nn.Module):

    def __init__(self, mode, overlap, chunk_width, gt_alignment_type, full_seq_sample_mode, use_random_sequences = True, use_subsampled_points_for_full_seq_metrics = True, max_points_for_icp_batch = 250000, max_points_for_icp_full_seq = 500000, trajectory_metrics = None, reconstruction_metrics = None, visualize = False, save_for_visualization = False, **kwargs):
        super().__init__()
        
        self.mode = mode
        self.num_overlap = overlap
        self.chunk_width = chunk_width
        self.full_seq_sample_mode = full_seq_sample_mode #full_seq_sample_mode= "chunk_overlap",
        self.gt_alignment_type = gt_alignment_type
        self.use_random_sequences = use_random_sequences
        self.use_subsampled_points_for_full_seq_metrics = use_subsampled_points_for_full_seq_metrics
        self.max_points_for_icp_batch = max_points_for_icp_batch
        self.max_points_for_icp_full_seq = max_points_for_icp_full_seq
        self.wandb_logger = None
        self.visualize = visualize
        self.save_for_visualization = save_for_visualization

        #initialize metrics
        self.trajectory_metrics = []
        if trajectory_metrics:
            for metric_dict in trajectory_metrics:
                metric = instantiate(metric_dict)
                self.trajectory_metrics.append(metric)

        self.reconstruction_metrics = []
        if reconstruction_metrics:
            for metric_dict in reconstruction_metrics:
                metric = instantiate(metric_dict)
                self.reconstruction_metrics.append(metric)
            

    def forward(self, predictions, batch, model, trainer) -> torch.Tensor:

        #grab wandb logger on rank zero
        if self.wandb_logger is None and trainer.is_global_zero:

            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    #save logger and create logging directory for plots
                    self.wandb_logger = logger

                    self.img_path = f"{self.wandb_logger.save_dir}/{self.wandb_logger.name}/{self.wandb_logger.version}/"

                    if not os.path.exists(self.img_path):
                        os.makedirs(self.img_path)

                    break
        
        batch_metrics_dict = {}
        seq_metrics_dict = {}

        #check if we are able to compute any metrics at all
        if len(self.trajectory_metrics) > 0 or (len(self.reconstruction_metrics) > 0 and ("world_points" in predictions or "depth" in predictions)):

            batch_metrics_dict = self.compute_batch_metrics(predictions,batch)

            if trainer.is_global_zero:

                if self.mode == "train" or self.mode == "validate":
                    datasets = trainer.val_dataloaders.dataset.base_dataset.datasets
                elif self.mode == "test":
                    datasets = trainer.test_dataloaders.dataset.base_dataset.datasets
                
                seq_metrics_dict = self.compute_full_sequence_metrics(datasets, model, predictions["pose_enc"].device)
            trainer.strategy.barrier("Compute sequence metrics")

        if self.visualize:
            if trainer.is_global_zero:
                #take first sequence only for visualization
                if self.mode == "train" or self.mode == "validate":
                    dataset = trainer.val_dataloaders.dataset.base_dataset.datasets[0]
                elif self.mode == "test":
                    dataset = trainer.test_dataloaders.dataset.base_dataset.datasets[0]
                
                self.visualize_sequence(dataset, model, predictions["pose_enc"].device)
            trainer.strategy.barrier("Visualize sequence")

        return batch_metrics_dict, seq_metrics_dict
    
    def compute_batch_metrics(self, predictions, batch):
        batch_metric_dict = {}
        
        log_additional_data(predictions,batch_metric_dict, self.chunk_width, self.num_overlap)

        pred_poses, gt_poses, pred_points, gt_points = self.prepare_data_for_metrics(predictions,batch,batch_metric_dict,max_points_icp=self.max_points_for_icp_batch)

        plot_title = f"seq: {batch['seq_name'][0]}"
        with torch.amp.autocast("cuda", enabled=False): #with torch.cuda.amp.autocast(enabled=False):
            if len(self.trajectory_metrics)>0:

                images = []
                #update metrics with per batch sequences
                for metric in self.trajectory_metrics:
                    #assure metric is on device:
                    metric = metric.to(gt_poses)
                    for pred,gt in zip(pred_poses, gt_poses):
                        metric.update(pred,gt)

                    batch_metric_dict.update(metric.compute())
                    metric.reset()

                    #plot metrics for first batch
                    if self.wandb_logger:
                        _, outpath = metric.plot(pred_poses[0],gt_poses[0],plot_title,f"{self.img_path}batch_")
                        if outpath: images.append(outpath)

                if len(images) > 0:
                    self.wandb_logger.log_image(key="val/batch_traj_metric_plots", images=images)

            if len(self.reconstruction_metrics) > 0 and ("world_points" in predictions or "depth" in predictions):

                images = []
                #update metrics with per batch sequences
                for metric in self.reconstruction_metrics:
                    #assure metric is on device:
                    metric = metric.to(gt_points[0])
                    for pred,gt in zip(pred_points, gt_points):
                        metric.update(pred,gt)

                    batch_metric_dict.update(metric.compute())
                    metric.reset()

                    #plot metrics for first batch
                    if self.wandb_logger:
                        _, outpath = metric.plot(pred_points[0],gt_points[0],plot_title,f"{self.img_path}batch_")
                        if outpath: images.append(outpath)

                if len(images) > 0:
                    self.wandb_logger.log_image(key="batch_metrics/batch_rec_metric_plots", images=images)

        return batch_metric_dict

    def compute_full_sequence_metrics(self, datasets, model, device):
        
        all_seq_metric_dict = {}

        sequences = gather_sequences(datasets, self.use_random_sequences)               
        
        for seq in sequences:            
            per_seq_dict = {}

            seq_data = get_sequence_data(*seq)
            seq_data["images"] = seq_data["images"].to(device)

            predictions = apply_model_to_sequence(seq_data, model, self.chunk_width, self.num_overlap, self.full_seq_sample_mode, self.gt_alignment_type)
            
            log_additional_data(predictions,per_seq_dict, self.chunk_width, self.num_overlap)

            pred_poses, gt_poses, pred_points, gt_points = self.prepare_data_for_metrics(predictions,seq_data,per_seq_dict,max_points_icp=self.max_points_for_icp_full_seq,output_subsampled_points=self.use_subsampled_points_for_full_seq_metrics,device=device)

            plot_title = f"{seq_data['dataset_name']}_seq[{seq_data['seq_name']}]"

            #adjust filename when saving plots, so we can save multiple sequence results
            if self.use_random_sequences:
                logging_prefix = "seq_metrics/"
                image_path = self.img_path
            else:
                logging_prefix = f"{seq_data['dataset_name']}_{seq_data['seq_name']}/"
                image_path = f"{self.img_path}[{seq_data['dataset_name']}_{seq_data['seq_name']}]_"

            if self.save_for_visualization:
                self.save_dict_for_visualization(predictions, seq_data, image_path)

            with torch.amp.autocast("cuda", enabled=False): #with torch.cuda.amp.autocast(enabled=False):
                if len(self.trajectory_metrics)>0:
                    assert pred_poses.shape[0]==1, "Only one sequence is processed at a time"

                    images = []
                    for metric in self.trajectory_metrics:
                        if self.wandb_logger:
                            #gather metrics and generate_plots
                            metrics, outpath = metric.plot(pred_poses[0],gt_poses[0],plot_title,image_path)
                            if outpath: images.append(outpath)
                        else:
                            #only gather metrics
                            metrics, _ = metric.plot(pred_poses[0],gt_poses[0])
                        per_seq_dict.update(metrics)
                        
                    if len(images) > 0:
                        self.wandb_logger.log_image(key=f"{logging_prefix}seq_traj_metric_plots", images=images)
                            

                if len(self.reconstruction_metrics) > 0 and ("world_points" in predictions or "depth" in predictions):
                    assert len(pred_points) ==1, "Only one sequence is processed at a time"

                    images = []
                    for metric in self.reconstruction_metrics:
                        if self.wandb_logger:
                            #gather metrics and generate_plots
                            metrics, outpath = metric.plot(pred_points[0],gt_points[0],plot_title,image_path)
                            if outpath: images.append(outpath)
                        else:
                            #only gather metrics
                            metrics, _ = metric.plot(pred_points[0],gt_points[0])
                        per_seq_dict.update(metrics)
                    
                    if len(images) > 0:
                        self.wandb_logger.log_image(key=f"{logging_prefix}seq_rec_metric_plots", images=images)

            #add prefix to dict keys
            for key,value in per_seq_dict.items():
                all_seq_metric_dict[f"{logging_prefix}{key}"] = value

        return all_seq_metric_dict
    

    def prepare_data_for_metrics(self, pred_dict, gt_dict, log_dict = None, valid_point_quantile = 0.25, max_points_icp = None, output_subsampled_points = False, device = None):
        
        with torch.amp.autocast("cuda", enabled=False): #with torch.cuda.amp.autocast(enabled=False):
            prepare_point_data = len(self.reconstruction_metrics) > 0 and ("world_points" in pred_dict or "depth" in pred_dict)
            prepare_pose_data = len(self.trajectory_metrics) > 0 # or (prepare_point_data and align_trajectories)

            if prepare_pose_data:
                B,S = gt_dict["extrinsics"].shape[:2]

                if pred_dict["pose_enc"].shape[-1] == 9:
                    #prepare poses
                    pred_extr, pred_intr = pose_encoding_to_extri_intri(pred_dict["pose_enc"],image_size_hw=gt_dict["images"].shape[-2:])
                elif pred_dict["pose_enc"].shape[-1] == 7:
                    pred_extr = pose_encoding_to_extri(pred_dict["pose_enc"])[:,:,:3,:4]

                    #use gt intrinsics for now
                    pred_intr = gt_dict["intrinsics"]
                else:
                    raise ValueError("Unkown pose enc type")

                #convert to 4x4 c2w matrices
                pred_poses = closed_form_inverse_se3(pred_extr.reshape(B*S,3,4)).reshape(B,S,4,4)
                gt_poses = closed_form_inverse_se3(gt_dict["extrinsics"].reshape(B*S,3,4)).reshape(B,S,4,4)

                #move to device if specified
                if device:
                    pred_poses = pred_poses.to(device)
                    gt_poses = gt_poses.to(device)

                if not prepare_point_data:
                    return pred_poses, gt_poses, None, None
                
            B, S, H, W, _ = gt_dict["world_points"].shape

            #prepare points
            #prioritize unprojected depths over point maps
            if "depth" in pred_dict:
                pred_points = unproject_depth_map_to_point_map(pred_dict["depth"],pred_extr,pred_intr)
                pred_points_conf = pred_dict["depth_conf"]
            else:
                pred_points = pred_dict["world_points"]
                pred_points_conf = pred_dict["world_points_conf"]

            #compute pred_point_mask
            pred_point_conf_treshold = torch_quantile(pred_points_conf,valid_point_quantile)
            pred_point_mask = pred_points_conf > pred_point_conf_treshold

            #subsample pred points if too many for icp
            valid_points_gt = gt_dict["point_masks"].sum().item()
            if max_points_icp and valid_points_gt > max_points_icp:
                #subsample 
                pre_subsample_point_mask_gt = gt_dict["point_masks"].unsqueeze(-1).permute(0,1,4,2,3).reshape(B * S, 1, H, W).float()

                #initially assume uniform distribution of valid points
                subsample_factor = ceil(np.sqrt(valid_points_gt / max_points_icp))

                last_subsample_factor = 0

                #exponential search
                while valid_points_gt > max_points_icp:
                    
                    if last_subsample_factor > 0:
                        last_subsample_factor = subsample_factor
                        subsample_factor = subsample_factor * 2
                    else:
                        last_subsample_factor = subsample_factor

                    #sanity check
                    if subsample_factor > max(H,W):
                        break

                    #finetune subsampling factor on gt mask (since pred are even fewer points)
                    gt_points_mask_subsampled = torch.nn.functional.interpolate(pre_subsample_point_mask_gt,size=(H//subsample_factor,W//subsample_factor),mode='bilinear',align_corners=False)
                    valid_points_gt = (gt_points_mask_subsampled > 0.5).sum().item()
                
                #binary search between lower and upper bound (only needed if out inital assumption is off)
                if last_subsample_factor != subsample_factor:
                    
                    while last_subsample_factor + 1 < subsample_factor:
                        interpolated_subsample_factor = (last_subsample_factor + subsample_factor) // 2

                        gt_points_mask_subsampled = torch.nn.functional.interpolate(pre_subsample_point_mask_gt,size=(H//interpolated_subsample_factor,W//interpolated_subsample_factor),mode='bilinear',align_corners=False)
                        interpolated_valid_points_gt = (gt_points_mask_subsampled > 0.5).sum().item()

                        if interpolated_valid_points_gt <= max_points_icp:
                            subsample_factor, valid_points_gt = interpolated_subsample_factor, interpolated_valid_points_gt
                        else:
                            last_subsample_factor = interpolated_subsample_factor

                #subsample points with final factor
                pred_points_subsampled = torch.nn.functional.interpolate(pred_points.permute(0,1,4,2,3).reshape(B * S, 3, H, W),size=(H//subsample_factor,W//subsample_factor),mode='bilinear',align_corners=False)
                pred_points_mask_subsampled = torch.nn.functional.interpolate(pred_point_mask.unsqueeze(-1).permute(0,1,4,2,3).reshape(B * S, 1, H, W).float(),size=(H//subsample_factor,W//subsample_factor),mode='bilinear',align_corners=False)
                gt_points_subsampled = torch.nn.functional.interpolate(gt_dict["world_points"].permute(0,1,4,2,3).reshape(B * S, 3, H, W),size=(H//subsample_factor,W//subsample_factor),mode='bilinear',align_corners=False)
                gt_points_mask_subsampled = torch.nn.functional.interpolate(pre_subsample_point_mask_gt,size=(H//subsample_factor,W//subsample_factor),mode='bilinear',align_corners=False)
                
                new_H, new_W = gt_points_mask_subsampled.shape[-2:]

                pred_points_subsampled = pred_points_subsampled.view(B,S,3,new_H, new_W).permute(0,1,3,4,2)
                pred_points_mask_subsampled = pred_points_mask_subsampled.view(B,S,1,new_H, new_W).permute(0,1,3,4,2).squeeze(-1) > 0.5
                gt_points_subsampled = gt_points_subsampled.view(B,S,3,new_H, new_W).permute(0,1,3,4,2)
                gt_points_mask_subsampled = gt_points_mask_subsampled.view(B,S,1,new_H, new_W).permute(0,1,3,4,2).squeeze(-1) > 0.5
            else:
                pred_points_subsampled = pred_points
                pred_points_mask_subsampled = pred_point_mask
                gt_points_subsampled = gt_dict["world_points"]
                gt_points_mask_subsampled = gt_dict["point_masks"]

            
            pred_points_masked = []
            gt_points_masked = []

            #prepare data for icp alignment:
            for b in range(B):

                masked_pred_points_batch = pred_points_subsampled[b][torch.logical_and(pred_points_mask_subsampled[b],gt_points_mask_subsampled[b])]
                masked_gt_points_batch = gt_points_subsampled[b][gt_points_mask_subsampled[b]]

                #move to device if specified
                if device:
                    masked_pred_points_batch = masked_pred_points_batch.to(device)
                    masked_gt_points_batch = masked_gt_points_batch.to(device)

                pred_points_masked.append(masked_pred_points_batch)
                gt_points_masked.append(masked_gt_points_batch)

            pred_pointclouds = Pointclouds(pred_points_masked)
            gt_pointclouds = Pointclouds(gt_points_masked)

            #do icp
            icp_solution = iterative_closest_point(pred_pointclouds,gt_pointclouds,max_iterations=30)

            pred_points = icp_solution.Xt.points_list()
            gt_points = gt_pointclouds.points_list()

            if not prepare_pose_data:
                return None, None, pred_points, gt_points
            else:
                return pred_poses, gt_poses, pred_points, gt_points
            
    def visualize_sequence(self, dataset, model, device): #self, seq_data, model):
        seq_data = get_sequence_data(*dataset)
        seq_data["images"] = seq_data["images"].to(device)
        predictions = apply_model_to_sequence(seq_data, model, self.chunk_width, self.num_overlap, self.full_seq_sample_mode, self.gt_alignment_type)
        
        if predictions["pose_enc"].shape[-1] == 9:
            #prepare poses
            pred_extr, pred_intr = pose_encoding_to_extri_intri(predictions["pose_enc"],image_size_hw=seq_data["images"].shape[-2:])
        elif predictions["pose_enc"].shape[-1] == 7:
            pred_extr = pose_encoding_to_extri(predictions["pose_enc"])[:,:,:3,:4]

            #use gt intrinsics for now
            pred_intr = seq_data["intrinsics"]
        else:
            raise ValueError("Unkown pose enc type")
        
        predictions["extrinsic"] = pred_extr
        predictions["intrinsic"] = pred_intr

        viser_server = viser_wrapper(predictions)

    def save_dict_for_visualization(self, predictions, seq_data, save_path):
        if predictions["pose_enc"].shape[-1] == 9:
            #prepare poses
            pred_extr, pred_intr = pose_encoding_to_extri_intri(predictions["pose_enc"],image_size_hw=seq_data["images"].shape[-2:])
        elif predictions["pose_enc"].shape[-1] == 7:
            pred_extr = pose_encoding_to_extri(predictions["pose_enc"])[:,:,:3,:4]

            #use gt intrinsics for now
            pred_intr = seq_data["intrinsics"]
        else:
            raise ValueError("Unkown pose enc type")
        
        predictions["extrinsic"] = pred_extr
        predictions["intrinsic"] = pred_intr

        out_dict = {}
        #convert to keys to numpy arrays and move to cpu
        for key in predictions:
            if key in ["images","intrinsic","extrinsic","world_points","world_points_conf","depth","depth_conf"]:
                if isinstance(predictions[key], torch.Tensor):
                    out_dict[key] = predictions[key].detach().cpu().numpy().squeeze(0)

        #save predictions for viser visualization later
        np.save(f"{save_path}visualization_data.npy", out_dict)

        
        out_dict_gt = {}
        #convert to keys to numpy arrays and move to cpu
        for key in seq_data:
            if key in ["images","intrinsics","extrinsics","world_points","point_masks","depths"]:
                if isinstance(seq_data[key], torch.Tensor):
                    out_dict_gt[key] = seq_data[key].detach().cpu().numpy().squeeze(0)

        #map gt keys to same naming convention as predictions
        out_dict_gt["intrinsic"] = out_dict_gt.pop("intrinsics")
        out_dict_gt["extrinsic"] = out_dict_gt.pop("extrinsics")
        out_dict_gt["world_points_conf"] = out_dict_gt.pop("point_masks").astype(float)
        out_dict_gt["depth_conf"] = out_dict_gt["world_points_conf"]
        out_dict_gt["depth"] = out_dict_gt.pop("depths")[...,None]
        
        #save predictions for viser visualization later
        np.save(f"{save_path}visualization_data_gt.npy", out_dict_gt)
        

def log_additional_data(pred_dict, log_dict, chunk_width, num_overlap):
    #log scales
    if "per_chunk_scales" in pred_dict:
        average_scale = torch.stack(pred_dict["per_chunk_scales"],dim=1).mean().item()
        log_dict["avg_per_chunk_scale"] = average_scale
    
    if "per_frame_scales" in pred_dict:
        average_scale = torch.cat(pred_dict["per_frame_scales"],dim=1).mean().item()
        log_dict["avg_per_frame_scale"] = average_scale

    if "alignment_scales_per_chunk" in pred_dict:
        average_scale = torch.stack(pred_dict["alignment_scales_per_chunk"],dim=1).mean().item()
        log_dict["avg_per_chunk_alignment_scale"] = average_scale

    if "alignment_scales" in pred_dict:
        average_scale = torch.tensor(pred_dict["alignment_scales"]).mean().item()
        log_dict["avg_alignment_scale"] = average_scale

    if "per_frame_pose_enc" in pred_dict:
        average_translation_norm = pred_dict["per_frame_pose_enc"][:,:,:3].norm(dim=-1).mean().item()
        log_dict["avg_per_frame_trans_norm"] = average_translation_norm

        quats = pred_dict["per_frame_pose_enc"][:,:,3:7]
        quats = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        
        average_rotation_magnitude = (2.0 * torch.sqrt(torch.clamp(1 - quats[:,:,-1]**2, min=0.0))).mean().item()
        log_dict["avg_per_frame_quat_magnitude"] = average_rotation_magnitude

    if "per_chunk_pose_enc" in pred_dict:
        average_translation_norm = pred_dict["per_chunk_pose_enc"][...,:3].norm(dim=-1).mean().item()
        log_dict["avg_per_chunk_trans_norm"] = average_translation_norm

        quats = pred_dict["per_chunk_pose_enc"][...,3:7]
        quats = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        
        average_rotation_magnitude = (2.0 * torch.sqrt(torch.clamp(1 - quats[:,-1]**2, min=0.0))).mean().item()
        log_dict["avg_per_chunk_quat_magnitude"] = average_rotation_magnitude

        if pred_dict["per_chunk_pose_enc"].shape[-1] == 8:
            average_scale = pred_dict["per_chunk_pose_enc"][...,7].mean().item()
            log_dict["avg_per_chunk_scale"] = average_scale


    if "global_tokens" in pred_dict:
        global_tokens_first = pred_dict["global_tokens"][0].cpu()
        global_tokens_last = pred_dict["global_tokens"][-1].cpu()
        B, N = global_tokens_last.shape[:2]

        global_norm_first = F.normalize(global_tokens_first, p=2, dim=-1, eps=1e-8)  # B x N x D
        global_norm_last = F.normalize(global_tokens_last, p=2, dim=-1, eps=1e-8)  # B x N x D

        cosine_sim = torch.bmm(global_norm_last, global_norm_last.transpose(1, 2))  # B x N x N
        mask = 1.0 - torch.eye(N, device=global_norm_last.device).unsqueeze(0)
        cosine_sim_offdiag = cosine_sim * mask
        avg_similarity = (cosine_sim_offdiag.sum() / (B * N * (N - 1))).item()
        log_dict["avg_global_token_similarity"] = avg_similarity

        cosine_sim = torch.bmm(global_norm_first, global_norm_first.transpose(1, 2))  # B x N x N
        mask = 1.0 - torch.eye(N, device=global_norm_first.device).unsqueeze(0)
        cosine_sim_offdiag = cosine_sim * mask
        avg_similarity = (cosine_sim_offdiag.sum() / (B * N * (N - 1))).item()
        log_dict["avg_global_token_similarity_first"] = avg_similarity

        pred_dict.pop("global_tokens", None)
    

def gather_sequences(datasets, use_random_sequences):
    sequences = []
    if use_random_sequences:
        #sample one random dataset sequence
        randomDatasetIdx = np.random.randint(0, len(datasets))
        randomDataset = datasets[randomDatasetIdx]

        random_seqIdx = np.random.randint(0, randomDataset.sequence_list_len)

        seq_name = randomDataset.get_seq_name(random_seqIdx)
        num_sequence_frames = randomDataset.seq_frame_num[random_seqIdx]

        sequences.append((randomDataset,random_seqIdx,seq_name,num_sequence_frames))
    else:
        #sample all sequences of all datasets
        for i in range(len(datasets)):
            dataset = datasets[i]

            for j in range(dataset.sequence_list_len):
                seq_name = dataset.get_seq_name(j)
                num_sequence_frames = dataset.seq_frame_num[j]
                sequences.append((dataset,j,seq_name,num_sequence_frames))

    return sequences

def get_sequence_data(dataset, seq_index, seq_name, seq_num_frames):
    
    #gather seq gt data
    seq_data = dataset.get_data(seq_index,-1,None,np.arange(seq_num_frames))
    
    # --- Data Conversion and Preparation ---
    # Convert numpy arrays to tensors
    images = torch.from_numpy(np.stack(seq_data["images"]).astype(np.float32)).contiguous()
    # Normalize images from [0, 255] to [0, 1]
    images = images.permute(0,3,1,2).to(torch.get_default_dtype()).div(255)

    # Convert other data to tensors with appropriate types
    depths = torch.from_numpy(np.stack(seq_data["depths"]).astype(np.float32))
    extrinsics = torch.from_numpy(np.stack(seq_data["extrinsics"]).astype(np.float32))
    intrinsics = torch.from_numpy(np.stack(seq_data["intrinsics"]).astype(np.float32))
    cam_points = torch.from_numpy(np.stack(seq_data["cam_points"]).astype(np.float32))
    world_points = torch.from_numpy(np.stack(seq_data["world_points"]).astype(np.float32))
    point_masks = torch.from_numpy(np.stack(seq_data["point_masks"])) # Mask indicating valid depths / world points / cam points per frame
    ids = torch.from_numpy(seq_data["ids"])    # Frame indices sampled from the original sequence

    #convert to first frame centered coordinate system
    with torch.amp.autocast("cuda", enabled=False): #with torch.cuda.amp.autocast(enabled=False):
        normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = \
                normalize_camera_extrinsics_and_points_batch(
                    extrinsics=extrinsics.unsqueeze(0),
                    cam_points=cam_points.unsqueeze(0),
                    world_points=world_points.unsqueeze(0),
                    depths=depths.unsqueeze(0),
                    point_masks=point_masks.unsqueeze(0),
                    scale_by_points=False
                )

    return {
            "dataset_name": dataset.__class__.__name__,
            "seq_name": seq_name,
            "ids": ids.unsqueeze(0),
            "images": images.unsqueeze(0),
            "depths": normalized_depths,
            "extrinsics": normalized_extrinsics,
            "intrinsics": intrinsics.unsqueeze(0),
            "cam_points": normalized_cam_points,
            "world_points": normalized_world_points,
            "point_masks": point_masks.unsqueeze(0),
            }
    
        
def apply_model_to_sequence(batch, model, chunk_width, num_overlap, sample_mode, alignment_type):
    """Generates model outputs on given image sequence.
        
    Args:
        batch: dict with gt data
        model: Model that generates outputs
        chunk_width: Number of images used for each image subsequence
        num_overlap: Number of overlapping frames between consecutive subsequences
    Returns:
        predictions: dict of outputs with batch dimension of 1 in each tensor
    """
    
    S = batch["images"].shape[1]
    
    #TODO optionally do random chunk_width sampling like in forward of model, for now only use max width
    chunk_width = chunk_width[0]
    num_overlap = num_overlap[0]

    #prepare chunk indices
    indices = generate_chunks(S, sample_mode, chunk_width, num_overlap)
    
    chunked_batch = chunk_batch(batch,indices)

    per_chunk_times = []

    #get model predictions
    last_chunk_outputs = None
    for i in range(len(indices)):
        
        per_chunk_time_start = time.time()

        gt_poses = chunked_batch["extrinsics"][i] if sample_mode == "chunk_gt" or sample_mode == "two_chunks" else None
        with torch.no_grad():
            predictions = model(chunked_batch['images'][i],num_overlap,last_chunk_outputs,True, gt_poses = gt_poses)

        per_chunk_times.append(time.time() - per_chunk_time_start)

        #move last items to cpu to save gpu memory
        moveDictListItemToCPU(predictions,-2)

        last_chunk_outputs = predictions

    moveDictListItemToCPU(predictions,-1)
    
    #perform alignment if necessary and convert to tensors
    alignAndConvertOutputs(predictions,batch,chunked_batch, alignment_type, chunk_width, num_overlap)

    return predictions

def torch_quantile(
    input,
    q,
    dim = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Validate out parameter
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out