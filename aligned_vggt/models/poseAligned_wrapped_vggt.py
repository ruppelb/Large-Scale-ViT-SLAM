import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.vggt.models.aggregator import Aggregator
from vggt.vggt.heads.camera_head import CameraHead
from vggt.vggt.heads.dpt_head import DPTHead
from vggt.vggt.heads.track_head import TrackHead
from vggt.vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from vggt.vggt.utils.geometry import closed_form_inverse_se3

from aligned_vggt.utils.alignment import scale_lse_solver
from aligned_vggt.utils.data import extri_to_pose_encoding, pose_encoding_to_extri
from aligned_vggt.utils.geometry import averagePoseEncodings

class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        
        super().__init__()

        self.intermediate_layer_indices = [4, 11, 17, 23] # indices of intermediate layers
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1", intermediate_layer_idx=range(len(self.intermediate_layer_indices))) if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1", intermediate_layer_idx=range(len(self.intermediate_layer_indices))) if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    def set_config(self, cfg: dict):
        self.camera_head = self.camera_head if cfg.enable_camera else None
        self.point_head = self.point_head if cfg.enable_point else None
        self.depth_head = self.depth_head if cfg.enable_depth else None
        self.track_head = self.track_head if cfg.enable_track else None

    def forward(self, images: torch.Tensor, num_overlap: int, context: dict=None, gt_poses: torch.Tensor = None) -> dict:
        """
        Forward pass of the pose-aligned VGGT model.
        Args:
            images (torch.Tensor): Input images of shape (B, S, C, H, W), where B is batch size, S is sequence length,
                                   C is number of channels, H and W are height and width of the images.
            num_overlap (int): Number of overlapping frames between consecutive chunks.
            context (dict, optional): Context dictionary containing information from previous chunks. Defaults to None.
            gt_poses (torch.Tensor, optional): Ground truth poses for alignment. Defaults to None. Only needed if sample
                                               mode is chunk_gt, in which case predictions will be aligned 
                                               using the ground truth poses.
        Returns:
            dict: A dictionary containing the following predictions:
                - 'pose_enc': List of pose encodings of all processed chunks.
                - 'depth': List of depth maps for all processed chunks.
                - 'depth_conf': List of depth confidence maps for all processed chunks.
                - 'world_points': List of 3D point maps for all processed chunks.
                - 'world_points_conf': List of 3D point confidence maps for all processed chunks
                - 'images': List of input images for all processed chunks (only during inference).
        """

        B, S, C, H, W = images.shape

        predictions = {}
        
        # extract features
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        filtered_aggregated_tokens_list = [aggregated_tokens_list[i] for i in self.intermediate_layer_indices]
        aggregated_tokens_list.clear()
        del aggregated_tokens_list
        torch.cuda.empty_cache()

        # decode poses, depth, and points maps for each chunk separately
        with torch.amp.autocast("cuda", enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(filtered_aggregated_tokens_list)
                
                # only update last pose encoding, since only this is used during inference
                extr, intr = pose_encoding_to_extri_intri(pose_enc_list[-1],image_size_hw=images.shape[-2:])
                extr = torch.nn.functional.pad(extr, (0,0,0,1,0,0,0,0), mode="constant")
                extr[:, :, 3, 3] = 1.

                # assure first pose is identity
                extr_identity_alignment = closed_form_inverse_se3(extr[:,0]) # (B, 4, 4)
                point_identity_alignment = extr[:,0].detach().clone()
                extr = extr @ extr_identity_alignment.view(B,1,4,4)

                # ground truth scale alignment
                if gt_poses is not None:
                    # pad to 4x4
                    gt_poses = torch.nn.functional.pad(gt_poses, (0,0,0,1,0,0,0,0), mode="constant")
                    gt_poses[:, :, 3, 3] = 1.

                    if S > 1:
                        centering_transform = closed_form_inverse_se3(gt_poses[:,0]) # (B, 4, 4)

                        gt_poses_firstFrameCentered = gt_poses @ centering_transform.view(B,1,4,4)

                        # compute a gt scale alignment
                        gt_positions = gt_poses_firstFrameCentered[...,:3,3].cpu().numpy()
                        pred_positions = extr[...,:3,3].detach().cpu().numpy()

                        batch_scales = []
                        for b in range(B):
                            batch_scale = scale_lse_solver(pred_positions[b],gt_positions[b])

                            extr[b,:,:3,3] *= batch_scale

                            batch_scales.append(batch_scale)

                # compute chunk alignment if context is given
                if context is not None:
                    if gt_poses is not None:
                        mean_camera_transform = gt_poses[:,:1].to(extr) # (B, 1, 4, 4)
                    else:
                        context_overlapping = pose_encoding_to_extri(context["pose_enc"][-1][:, -num_overlap:]) # (B, N_overlap, 4, 4)

                        inv_overlapping = closed_form_inverse_se3(extr[:, :num_overlap].reshape(B*num_overlap,4,4)).reshape(B,num_overlap,4,4)

                        camera_transforms = inv_overlapping @ context_overlapping

                        if num_overlap > 1:
                            camera_transforms = extri_to_pose_encoding(camera_transforms)

                            mean_camera_transform = averagePoseEncodings(camera_transforms) # (B, 1, 7)

                            mean_camera_transform = pose_encoding_to_extri(mean_camera_transform) # (B, 1, 4, 4)
                        else:
                            mean_camera_transform = camera_transforms
                else:
                    mean_camera_transform = torch.eye(4, device=extr.device, dtype=extr.dtype).view(1,1,4,4).expand(B,-1,-1,-1)
                
                # apply final pose alignment
                aligned_extr = torch.matmul(extr, mean_camera_transform)
                aligned_pose_enc = extri_intri_to_pose_encoding(aligned_extr, intr, image_size_hw=images.shape[-2:])
                
                if context is None:
                    predictions["pose_enc"] = [aligned_pose_enc]
                else:
                    context.setdefault("pose_enc", []).append(aligned_pose_enc)
                    predictions["pose_enc"] = context["pose_enc"]

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    filtered_aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )

                # apply gt alignment scale
                if gt_poses is not None:
                    for b in range(B):
                        depth[b,...] *= batch_scales[b]

                if context is None:
                    predictions["depth"] = [depth]
                    predictions["depth_conf"] = [depth_conf]
                else:
                    # no need to align depth maps, since they are already aligned by the alignment head
                    context.setdefault("depth", []).append(depth)
                    predictions["depth"] = context["depth"]
                    context.setdefault("depth_conf", []).append(depth_conf)
                    predictions["depth_conf"] = context["depth_conf"]

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    filtered_aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )

                # align point maps using last stage transform pose enc
                if self.camera_head is not None:
                    
                    if gt_poses is not None:
                        # apply gt alignment scale
                        for b in range(B):
                            pts3d[b,...] *= batch_scales[b]

                    if context is not None:
                        point_transform = closed_form_inverse_se3(mean_camera_transform.squeeze(1)).unsqueeze(1)  # (B, 1, 4, 4)
                        
                        # apply identity alignment
                        point_transform = point_transform @ point_identity_alignment.view(B,1,4,4)
                    else:
                        point_transform = point_identity_alignment.view(B,1,4,4)
                    
                    # convert points to homogeneous coordinates
                    pts3d_h = torch.cat([pts3d, torch.ones_like(pts3d[..., :1])], dim=-1) # (B, SC, H, W, 4)

                    pts3d_h = pts3d_h.view(B, -1, 4)  # (B, SC*H*W, 4)
                    point_transform = point_transform.expand(-1, pts3d_h.size(1), -1, -1)  # (B, SC*H*W, 4, 4)

                    pts3d_transformed = torch.matmul(point_transform, pts3d_h.unsqueeze(-1)).squeeze(-1)  # (B, SC*H*W, 4)

                    pts3d = pts3d_transformed[..., :3].view(B,S,H,W,3) # (B, SC, H, W, 3)

                if context is None:
                    predictions["world_points"] = [pts3d]
                    predictions["world_points_conf"] = [pts3d_conf]
                else:
                    context.setdefault("world_points", []).append(pts3d)
                    predictions["world_points"] = context["world_points"]
                    context.setdefault("world_points_conf", []).append(pts3d_conf)
                    predictions["world_points_conf"] = context["world_points_conf"]

            if not self.training:
                if context is None:
                    predictions["images"] = [images] # store the images for visualization during inference
                else:
                    context.setdefault("images", []).append(images)
                    predictions["images"] = context["images"]

        return predictions