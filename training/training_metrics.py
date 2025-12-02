import torch
import torch.nn.functional as F
import numpy as np
import os
import time
from math import ceil, floor

from pytorch3d.structures import Pointclouds
from pytorch3d.ops import iterative_closest_point
from pytorch3d.ops.points_alignment import _apply_similarity_transform

from lightning.pytorch.loggers import WandbLogger

from hydra.utils import instantiate
from dataclasses import dataclass
from vggt.utils.pose_enc import extri_intri_to_pose_encoding,pose_encoding_to_extri_intri
from vggt.utils.rotation import quat_to_mat, mat_to_quat
from vggt.utils.geometry import closed_form_inverse_se3
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from aligned_vggt.utils.geometry import unproject_depth_map_to_point_map
from aligned_vggt.utils.visualization import viser_wrapper

#from eval.trajectory_metrics import eval_ate, eval_rpe
#from aligned_vggt.utils.alignment import scale_lse_solver, umeyama, apply_sim3_alignment_on_poses, apply_sim3_alignment_on_point_maps
from aligned_vggt.utils.data import alignAndConvertOutputs, generate_chunks, chunk_batch, moveDictListItemToCPU, pose_encoding_to_extri

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

        #add indicator that metrics are batch metrics
        #batch_metric_dict = { f"batch_{key}" : value for (key, value) in batch_metric_dict.items()}

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
            
            """
            #subsample point cloud
            if max_points_icp:
                pred_pointclouds_subsampled = pred_pointclouds.subsample(max_points=max_points_icp)
                gt_pointclouds_subsampled = gt_pointclouds.subsample(max_points=max_points_icp)
            else:
                pred_pointclouds_subsampled = pred_pointclouds
                gt_pointclouds_subsampled = gt_pointclouds
            """

            #do icp
            icp_solution = iterative_closest_point(pred_pointclouds,gt_pointclouds,max_iterations=30)

            """
            if not icp_solution.converged:
                #print("ICP did not converge")
                pass
            

            if output_subsampled_points:
                #directly use transformed_result
                pred_points = icp_solution.Xt.points_list()
                gt_points = gt_pointclouds.points_list()
            else:
                #apply transform
                rot,trans,s = icp_solution.RTs
                pred_points_padded = pred_pointclouds.points_padded()
                pred_points_padded = _apply_similarity_transform(pred_points_padded,rot,trans,s)
                pred_pointclouds = pred_pointclouds.update_padded(pred_points_padded)

                pred_points = pred_pointclouds.points_list()
                gt_points = gt_pointclouds.points_list()
            """

            pred_points = icp_solution.Xt.points_list()
            gt_points = gt_pointclouds.points_list()
            
            """
            #do per batch icp alignment
            for b in range(B):
                #mask point maps
                gt_points_batch_masked = gt_dict["world_points"][b][gt_dict["point_masks"][b]]
                pred_points_batch_masked = pred_points[b][torch.logical_and(pred_point_mask[b],gt_dict["point_masks"][b])]

                
                #prepare data for o3d
                gt_points_batch_masked = gt_points_batch_masked.detach().cpu().numpy().reshape(-1,3)
                pred_points_batch_masked = pred_points_batch_masked.detach().cpu().float().numpy().reshape(-1,3)

                #run icp to get even better alignment (on downsampled overall points) -> apply icp alignment
                gt_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_points_batch_masked))
                pred_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred_points_batch_masked))

                voxel_size = 0.1 #0.05
                gt_pc_downsampled = gt_pc.voxel_down_sample(voxel_size)
                pred_pc_downsampled = pred_pc.voxel_down_sample(voxel_size)
                
                registered_points = o3d.pipelines.registration.registration_icp(pred_pc_downsampled, gt_pc_downsampled, max_correspondence_distance=0.1, init=np.eye(4), estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())

                # Apply the transformation to the full-resolution source point cloud
                pred_pc.transform(registered_points.transformation)

                pred_points_aligned.append(np.asarray(pred_pc.points))
                gt_points.append(gt_points_batch_masked)
            

            pred_points = torch.stack([torch.from_numpy(item).float().to(device) for item in pred_points_aligned])
            gt_points = torch.stack([torch.from_numpy(item).float().to(device) for item in gt_points])
            """

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

    if "global_scales" in pred_dict:
        average_scale = pred_dict["global_scales"].mean().item()
        log_dict["avg_global_scale"] = average_scale

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

    if "per_frame_pose_enc_list" in pred_dict:
        average_scale = torch.cat([item[:,:,7:8] for item in pred_dict["per_frame_pose_enc_list"]],dim=0).mean().item()
        log_dict["avg_per_frame_scale"] = average_scale
        average_translation_norm = torch.cat([item[:,:,:3] for item in pred_dict["per_frame_pose_enc_list"]],dim=0).norm(dim=-1).mean().item()
        log_dict["avg_per_frame_trans_norm"] = average_translation_norm
        #average_rotation_quat_norm = torch.cat([item[:,:,3:7] for item in pred_dict["per_frame_pose_enc_list"]],dim=0).norm(dim=-1).mean().item()   
        #ws = torch.cat([item[:,:,6] for item in pred_dict["per_frame_pose_enc_list"]],dim=0)
        quats = torch.cat([item[:,:,3:7] for item in pred_dict["per_frame_pose_enc_list"]],dim=0)
        quats = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        
        average_rotation_magnitude = (2.0 * torch.sqrt(torch.clamp(1 - quats[:,-1]**2, min=0.0))).mean().item()
        log_dict["avg_per_frame_quat_magnitude"] = average_rotation_magnitude
    
    if "per_chunk_pose_enc_list" in pred_dict:
        average_scale = torch.cat([item[:,7:8] for item in pred_dict["per_chunk_pose_enc_list"]],dim=0).mean().item()
        log_dict["avg_per_chunk_scale"] = average_scale
        average_translation_norm = torch.cat([item[:,:3] for item in pred_dict["per_chunk_pose_enc_list"]],dim=0).norm(dim=-1).mean().item()
        log_dict["avg_per_chunk_trans_norm"] = average_translation_norm
        #average_rotation_quat_norm = torch.cat([item[:,3:7] for item in pred_dict["per_chunk_pose_enc_list"]],dim=0).norm(dim=-1).mean().item()
        #ws = torch.cat([item[:,6] for item in pred_dict["per_chunk_pose_enc_list"]],dim=0)
        quats = torch.cat([item[:,3:7] for item in pred_dict["per_chunk_pose_enc_list"]],dim=0)
        quats = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        
        average_rotation_magnitude = (2.0 * torch.sqrt(torch.clamp(1 - quats[:,-1]**2, min=0.0))).mean().item()
        log_dict["avg_per_chunk_quat_magnitude"] = average_rotation_magnitude

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
    
    if "memory_token_list" in pred_dict:
        S = len(pred_dict["memory_token_list"])
        memory_tokens_first = pred_dict["memory_token_list"][0].cpu()
        memory_tokens_mid = pred_dict["memory_token_list"][S//2].cpu()
        memory_tokens_last = pred_dict["memory_token_list"][-1].cpu()
        B, N = memory_tokens_last.shape[:2]
        
        mem_norm_first = F.normalize(memory_tokens_first, p=2, dim=-1, eps=1e-8)  # B x N x D
        mem_norm_mid = F.normalize(memory_tokens_mid, p=2, dim=-1, eps=1e-8)  # B x N x D
        mem_norm_last = F.normalize(memory_tokens_last, p=2, dim=-1, eps=1e-8)  # B x N x D

        #cosine between different last layer tokens
        cosine_sim = torch.bmm(mem_norm_last, mem_norm_last.transpose(1, 2))  # B x N x N
        mask = 1.0 - torch.eye(N, device=memory_tokens_last.device).unsqueeze(0)
        cosine_sim_offdiag = cosine_sim * mask
        avg_similarity = (cosine_sim_offdiag.sum() / (B * N * (N - 1))).item()
        log_dict["avg_memory_token_similarity"] = avg_similarity

        #cosine between different mid layer tokens
        cosine_sim = torch.bmm(mem_norm_mid, mem_norm_mid.transpose(1, 2))  # B x N x N
        mask = 1.0 - torch.eye(N, device=memory_tokens_last.device).unsqueeze(0)
        cosine_sim_offdiag = cosine_sim * mask
        avg_similarity = (cosine_sim_offdiag.sum() / (B * N * (N - 1))).item()
        log_dict["avg_memory_token_similarity_mid"] = avg_similarity

        #cosine between same tokens temporally
        cosine_sim = torch.bmm(mem_norm_first, mem_norm_last.transpose(1, 2))  # B x N x N
        cosine_sim_offdiag = cosine_sim * torch.eye(N, device=memory_tokens_last.device).unsqueeze(0)
        avg_similarity = (cosine_sim_offdiag.sum() / (B * N)).item()
        log_dict["avg_memory_token_similarity_temporal"] = avg_similarity

        #compute memory update each step
        memory_tokens = torch.stack([p.cpu() for p in pred_dict["memory_token_list"]],dim=1) #B,S,N,D
        diff = memory_tokens[:,1:] - memory_tokens[:,:-1] # B,S-1,N,D
        diff_norm = diff.norm(dim=-1) #B,S-1,N
        diff_avg = diff_norm.mean(dim=(0,2)) #S-1
        print(f"Average memory updates per frame: \n {diff_avg}")

        pred_dict.pop("memory_token_list",None)

    if "memory_gate_list" in pred_dict:
        memory_gates_per_frame = pred_dict["memory_gate_list"][0]
        average_memory_gates_per_frame = (torch.stack([p.cpu() for p in memory_gates_per_frame],dim=1)).mean(dim=(0,2,3))
        print(f"Average memory gates per frame: \n {average_memory_gates_per_frame}")

        
        frame_gates_per_frame = pred_dict["memory_gate_list"][1]
        average_frame_gates_per_frame = (torch.stack([p.cpu() for p in frame_gates_per_frame],dim=1)).mean(dim=(0,2,3))
        print(f"Average frame gates per frame: \n {average_frame_gates_per_frame}")
        

        pred_dict.pop("memory_gate_list",None)

    """
    if "memory_conf_list" in pred_dict:
        conf_per_frame = pred_dict["memory_conf_list"][0]
        average_conf_per_frame = (torch.stack([p.cpu() for p in conf_per_frame],dim=1)).mean(dim=(0,2,3))
        print(f"Average conf per frame: \n {average_conf_per_frame}")

        mem_conf_per_frame = pred_dict["memory_conf_list"][1]
        average_mem_conf_per_frame = (torch.stack([p.cpu() for p in mem_conf_per_frame],dim=1)).mean(dim=(0,2,3))
        print(f"Average mem conf per frame: \n {average_mem_conf_per_frame}")

        pred_dict.pop("memory_conf_list",None)
    """
    #log times
    if "full_sequence_inference_time" in pred_dict:
        log_dict["full_sequence_inference_time"] = pred_dict["full_sequence_inference_time"]
        log_dict["full_sequence_inference_fps"] = pred_dict["pose_enc"].shape[1] / log_dict["full_sequence_inference_time"]

    if "per_chunk_inference_time" in pred_dict:
        log_dict["avg_per_chunk_inference_time"] = np.mean(pred_dict["per_chunk_inference_time"])
        log_dict["avg_per_chunk_inference_fps"] = (chunk_width[0]-num_overlap[0]) / log_dict["avg_per_chunk_inference_time"]

    if "alignment_module_inference_time" in pred_dict:
        log_dict["avg_alignment_module_inference_time"] = np.mean(pred_dict["alignment_module_inference_time"])

    if "alignment_computation_inference_time" in pred_dict:
        log_dict["avg_alignment_computation_inference_time"] = np.mean(pred_dict["alignment_computation_inference_time"])
    

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
    #seq_gt_extr, _, _, _ = normalize_camera_extrinsics_and_points_batch(extrinsics=seq_gt_extr.unsqueeze(0),scale_by_points=False)
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

    full_seq_time_start = time.time()

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

    full_seq_time = time.time() - full_seq_time_start
    predictions["full_sequence_inference_time"] = full_seq_time
    predictions["per_chunk_inference_time"] = per_chunk_times
    
    
    #perform alignment if necessary and convert to tensors
    alignAndConvertOutputs(predictions,batch,chunked_batch, alignment_type, chunk_width, num_overlap)

    #delete keys we don't need
    predictions.pop("aggregated_tokens", None)
    predictions.pop("scale_tokens", None)
    #predictions.pop("global_tokens", None)
    predictions.pop("pose_enc_list",None)
    #predictions.pop("images",None)
    predictions.pop("overlapping_camera_transforms",None)
    predictions.pop("overlapping_relative_pose_pairs",None)
    predictions.pop("overlapping_depth_pairs",None)
    predictions.pop("overlapping_depth_conf_pairs",None)
    predictions.pop("overlapping_pts3d_pairs",None)
    predictions.pop("overlapping_pts3d_conf_pairs",None)
    predictions.pop("depth_pairs_ov",None)
    predictions.pop("depth_conf_pairs_ov",None)
    
    predictions.pop("cam_poses_rel",None)

    #alignment and conversion

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




"""
last_chunk_outputs = None
for j, chunk in enumerate(indices):
    #images = torch.index_select(seq_images, 1, torch.tensor(chunk,device=device).long())
    #images = images.unsqueeze(0) #add batch dimension
    
    with torch.no_grad():
        predictions = model(images,last_chunk_outputs,True)

    
    last_chunk_outputs = prediction.copy()

    #delete keys we don't need
    prediction.pop("aggregated_tokens", None)
    prediction.pop("pose_enc_list",None)
    prediction.pop("images",None)
    prediction.pop("overlapping_camera_transforms",None)
    prediction.pop("overlapping_relative_pose_pairs",None)
    prediction.pop("overlapping_depth_pairs",None)
    prediction.pop("overlapping_depth_conf_pairs",None)
    prediction.pop("overlapping_pts3d_pairs",None)
    prediction.pop("overlapping_pts3d_conf_pairs",None)

    #extrinsic, intrinsic = pose_encoding_to_extri_intri(prediction["pose_enc"], images.shape[-2:])
    #prediction["extrinsics"] = extrinsic
    #prediction["intrinsics"] = intrinsic

    if j==0:
        predictions = {}

        for key in prediction.keys():
            if isinstance(prediction[key], torch.Tensor):
                prediction[key] = prediction[key].detach() #.cpu()
            
            predictions[key] = [prediction[key]]
    else:
        #append the new predictions to the existing ones
        for key in prediction.keys():
            if isinstance(prediction[key], torch.Tensor):
                prediction[key] = prediction[key].detach() #.cpu()

            predictions[key].append(prediction[key][:,num_overlap:])
    

for key in predictions.keys():
    predictions[key] = torch.cat(predictions[key], axis=1)
"""

"""
def compute_full_sequence_metrics(trainer, model, wandb_logger, sample_mode, num_overlap, seq_width, gt_alignment_type, device):
    
    Computes the full dataset metric for the model.
    This method is used to compute the metric for the entire dataset, not just a single batch.
    

    dataset = trainer.val_dataloaders.dataset

    full_seq_dict = {}

    randomDatasetIdx = np.random.randint(0, len(dataset.base_dataset.datasets))
    randomDataset = dataset.base_dataset.datasets[randomDatasetIdx]

    random_seqIdx = np.random.randint(0, randomDataset.sequence_list_len)

    seq_name = randomDataset.get_seq_name(random_seqIdx)
    num_sequence_frames = randomDataset.seq_frame_num[random_seqIdx]

    #prepare chunk indices
    chunks = generate_sequences(num_sequence_frames, sample_mode, seq_width, num_overlap)

    #gather seq gt data
    seq_data = randomDataset.get_data(random_seqIdx,-1,None,np.arange(num_sequence_frames))
    
    seq_images = torch.from_numpy(np.stack(seq_data["images"]).astype(np.float32)).contiguous()
    seq_images = seq_images.permute(0,3,1,2).to(torch.get_default_dtype()).div(255).to(device) # Normalize images from [0, 255] to [0, 1]

    seq_gt_extr = torch.from_numpy(np.stack(seq_data["extrinsics"]).astype(np.float32))
    #convert to first frame centered coordinate system
    seq_gt_extr, _, _, _ = normalize_camera_extrinsics_and_points_batch(extrinsics=seq_gt_extr.unsqueeze(0),scale_by_points=False)
    #seq_gt_extr, _, _, _ = normalize_camera_extrinsics_and_points_batch(extrinsics=seq_gt_extr.unsqueeze(0),cam_points=torch.from_numpy(np.stack(seq_gt_data["cam_points"]).astype(np.float32)).unsqueeze(0),world_points=torch.from_numpy(np.stack(seq_gt_data["world_points"]).astype(np.float32)).unsqueeze(0),depths=torch.from_numpy(np.stack(seq_gt_data["depths"]).astype(np.float32)).unsqueeze(0),point_masks=torch.from_numpy(np.stack(seq_gt_data["point_masks"]).astype(np.float32)).unsqueeze(0))
    seq_gt_extr = seq_gt_extr.squeeze(0)
    
    #convert to 4x4 matrices
    seq_gt_extr = torch.nn.functional.pad(seq_gt_extr, (0,0,0,1,0,0), mode="constant")
    seq_gt_extr[:, 3, 3] = 1.
    
    last_chunk_outputs = None
    last_scale = 1.0
    for j, chunk in enumerate(chunks):
        #gather images
        #metadata = randomDataset.get_data(random_seqIdx,-1,None,np.array(chunk))

        #gt_extr =  torch.from_numpy(np.stack(metadata["extrinsics"]).astype(np.float32)).to(device)
        #images = torch.from_numpy(np.stack(metadata["images"]).astype(np.float32)).contiguous()
                
        images = torch.index_select(seq_images, 0, torch.tensor(chunk,device=device).long())
        images = images.unsqueeze(0) #add batch dimension
        
        predictions = model(images,last_chunk_outputs)

        pred_extr, _ = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:],build_intrinsics=False)

        pred_extr = pred_extr.squeeze(0)  # remove batch dimension

        #Convert to 4x4 matrices
        pred_extr = torch.nn.functional.pad(pred_extr, (0,0,0,1,0,0), mode="constant")
        pred_extr[:, 3, 3] = 1.
        
        if sample_mode == "chunk_gt":
            
            #get gt extr
            gt_extr = torch.index_select(seq_gt_extr, 0, torch.tensor(chunk,device=device).long())
            first_cam_extrinsic_inv = closed_form_inverse_se3(gt_extr[0].unsqueeze(0))
            gt_extr = torch.matmul(gt_extr, first_cam_extrinsic_inv)

            #assure first pose is identity
            pred_extr = pred_extr @ closed_form_inverse_se3(pred_extr[0].unsqueeze(0)).squeeze(0)

            #scale align predicted poses to ground truth poses
            if seq_width > 1 and images.shape[1] == seq_width:
                
                prediction_scale  = torch.max(torch.norm(pred_extr[:, :3, 3], dim=-1))
                world_scale = torch.max(torch.norm(gt_extr[:, :3, 3], dim=-1))
                #world_scale = torch.norm(gt_extr[-1, :3, 3],dim=-1)
                #prediction_scale = torch.norm(pred_extr[-1, :3, 3],dim=-1)

                # Compute the isometric scale factor
                scale = world_scale / prediction_scale
            else:
                scale = last_scale
            
            pred_extr[...,:3,3] *= scale
            last_scale = scale

        if j==0:
            seq_pred_extr = [pred_extr.float().cpu().numpy()]
        else:

            if sample_mode == "chunk_gt":
                #transform to world coordinates
                pred_extr = pred_extr @ seq_gt_extr[j*seq_width].to(device)

            elif sample_mode == "chunk_overlap":
                #Not necessary when using aligned vggt as alignment is done in model
                #alignment to previous predictions
                #gather overlapped,compute alignment transform (sim3 or only scale),apply transform
                pass

            seq_pred_extr.append(pred_extr[num_overlap:].float().cpu().numpy()) #only use non-overlapping frames for better accuracy
        
        last_chunk_outputs = predictions

    seq_pred_extr = np.concatenate(seq_pred_extr, axis=0)
    seq_gt_extr = seq_gt_extr.numpy()

    #convert to c->w
    seq_pred_extr = closed_form_inverse_se3(seq_pred_extr)
    seq_gt_extr = closed_form_inverse_se3(seq_gt_extr)

    #|---> Trajectory Alignment <---|
    if sample_mode == "chunk_overlap":
        
        if gt_alignment_type == "scale_from_poses" or gt_alignment_type == None:
            scale = scale_lse_solver(seq_pred_extr[:,:3,3],seq_gt_extr[:,:3,3])
            seq_pred_extr[:,:3,3] *= scale
            full_seq_dict["seq_scale"] = scale

            #print(f"\nFull seq scale {scale}")
            #print(f"\n{closed_form_inverse_se3(seq_pred_extr[None, 1]).squeeze(0)}")
        elif gt_alignment_type == "sim3_from_poses":
            #TODO: change alignment methods to be the same in alignment_helper and metric script. (currently method of horn here and umeyama in alignment_helper)
            #TODO: correct application of transform (transform also has to be applied to rotations, otherwise RPE will be calculated wrong)
            rot,trans,scale = umeyama(seq_pred_extr[:,:3,3],seq_gt_extr[:,:3,3])
            seq_pred_extr[:,:3,3] = scale * (rot @ seq_pred_extr[:,:3,3,None]).squeeze(-1) + trans
            full_seq_dict["seq_scale"] = scale
    

    if wandb_logger is None:
        print("No WandbLogger found, skipping image logging.")
        ate_rmse, _, _, _, _, _= eval_ate(seq_pred_extr,seq_gt_extr,verbose=False)
        trans_rmse, _, _, _, _, _, rot_rmse, _, _, _, _, _ = eval_rpe(seq_pred_extr,seq_gt_extr,verbose=False)
    else:
        img_path = f"{wandb_logger.save_dir}/{wandb_logger.name}/{wandb_logger.version}"

        if not os.path.exists(img_path):
            os.makedirs(img_path)

        ate_rmse, _, _, _, _, _= eval_ate(seq_pred_extr,seq_gt_extr,f"{randomDataset.__class__.__name__}_seq[{seq_name}]",f"{img_path}/full_seq_ate",verbose=False)
        trans_rmse, _, _, _, _, _, rot_rmse, _, _, _, _, _ = eval_rpe(seq_pred_extr,seq_gt_extr,f"{randomDataset.__class__.__name__}_seq[{seq_name}]",f"{img_path}/full_seq_rpe",verbose=False)
        
        #log images
        wandb_logger.log_image(key="val/seq_metric_plots", images=[f"{img_path}/full_seq_ate_plot.png", f"{img_path}/full_seq_rpe_plot.png"])

    #log metrics
    full_seq_dict["seq_ate_rmse"] = ate_rmse
    full_seq_dict["seq_trans_rmse"] = trans_rmse
    full_seq_dict["seq_rot_rmse"] = rot_rmse
    

    return full_seq_dict

def compute_batch_metrics(predictions, batch):
    batch_metric_dict = {}

    #compute ate and rpe metrics for single chunks
    pred_extr, _ = pose_encoding_to_extri_intri(predictions["pose_enc"],build_intrinsics=False)
    
    #convert to 4x4
    pred_extr = torch.nn.functional.pad(pred_extr, (0, 0, 0, 1, 0, 0, 0, 0), mode="constant")
    pred_extr[..., 3, 3] = 1.
    gt_extr = torch.nn.functional.pad(batch["extrinsics"], (0, 0, 0, 1, 0, 0, 0, 0), mode="constant")
    gt_extr[..., 3, 3] = 1.
    
    pred_extr = pred_extr.cpu().numpy()
    gt_extr = gt_extr.cpu().numpy()

    ate_rmse = np.zeros((gt_extr.shape[0],), dtype=np.float32)
    trans_rmse = np.zeros((gt_extr.shape[0],), dtype=np.float32)
    rot_rmse = np.zeros((gt_extr.shape[0],), dtype=np.float32)
    for i in range(gt_extr.shape[0]):
        #convert to c->w
        pred_extr[i] = closed_form_inverse_se3(pred_extr[i])
        gt_extr[i] = closed_form_inverse_se3(gt_extr[i])
        
        ate_rmse[i], _, _, _, _, _= eval_ate(pred_extr[i],gt_extr[i],verbose=False)
        trans_rmse[i], _, _, _, _, _, rot_rmse[i], _, _, _, _, _ = eval_rpe(pred_extr[i],gt_extr[i],verbose=False)

    batch_metric_dict["ate_rmse"] = ate_rmse.mean()
    batch_metric_dict["trans_rmse"] = trans_rmse.mean()
    batch_metric_dict["rot_rmse"] = rot_rmse.mean()

    return batch_metric_dict


def extri_to_pose_encoding(
    extrinsics
):

    # extrinsics: BxSx3x4
    R = extrinsics[:, :, :3, :3]  # BxSx3x3
    T = extrinsics[:, :, :3, 3]  # BxSx3

    quat = mat_to_quat(R)

    pose_encoding = torch.cat([T, quat], dim=-1).float()

    return pose_encoding


def pose_encoding_to_extri(
    pose_encoding
): 
    #pose enc: BxSx9
    T = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]
    R = quat_to_mat(quat)
    extrinsics = torch.cat([R, T[..., None]], dim=-1)
    extrinsics = torch.nn.functional.pad(extrinsics, (0,0,0,1,0,0,0,0), mode="constant")
    extrinsics[:,:, 3, 3] = 1.

    return extrinsics
"""