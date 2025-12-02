# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead

from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from vggt.utils.rotation import quat_to_mat, mat_to_quat
from vggt.utils.geometry import closed_form_inverse_se3

from aligned_vggt.utils.alignment import scale_lse_solver


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        
        super().__init__()

        self.intermediate_layer_indices = [4, 11, 17, 23]  # indices of intermediate layers
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1", intermediate_layer_idx=range(len(self.intermediate_layer_indices))) if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1", intermediate_layer_idx=range(len(self.intermediate_layer_indices))) if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

        #self.num_overlap = num_overlap

    def set_config(self, cfg: dict):
        #self.num_overlap = cfg.num_overlap
        self.camera_head = self.camera_head if cfg.enable_camera else None
        self.point_head = self.point_head if cfg.enable_point else None
        self.depth_head = self.depth_head if cfg.enable_depth else None
        self.track_head = self.track_head if cfg.enable_track else None

    def forward(self, images: torch.Tensor, num_overlap, context : dict = None, mergeResults : bool = False, gt_poses : torch.Tensor =None):
        #context is a dict with lists or None for the first chunk

        B, S, C, H, W = images.shape

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        filtered_aggregated_tokens_list = [aggregated_tokens_list[i] for i in self.intermediate_layer_indices]
        aggregated_tokens_list.clear()
        del aggregated_tokens_list
        torch.cuda.empty_cache()

        predictions = {}

        if not self.training:
            if context is None:
                predictions["images"] = [images]  # store the images for visualization during inference
            else:
                context.setdefault("images", []).append(images)
                predictions["images"] = context["images"]

        #decode poses, depth, points maps, and tracks for each chunk seperately
        with torch.amp.autocast("cuda", enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(filtered_aggregated_tokens_list)

                if not self.training:
                    start_time = time.time()
                
                if self.training:
                    #align all pose encodings during training, so we can compute multi-stage camera loss
                    for i in range(len(pose_enc_list)):
                        new_pose_enc, point_identity_alignment, mean_camera_transform, batch_scales = align_poses(pose_enc_list[i], images.shape[-2:], num_overlap, gt_poses = gt_poses, context_pose_enc=context["pose_enc_list"][-1][i] if context is not None else None)
                        pose_enc_list[i] = new_pose_enc
                    
                else:
                    new_pose_enc, point_identity_alignment, mean_camera_transform, batch_scales = align_poses(pose_enc_list[-1],images.shape[-2:], num_overlap, gt_poses = gt_poses, context_pose_enc=context["pose_enc_list"][-1][-1] if context is not None else None)
                    pose_enc_list[-1] = new_pose_enc
                
                if not self.training:
                    if context is None:
                        predictions["alignment_computation_inference_time"] = [time.time() - start_time]
                    else:
                        context.setdefault("alignment_computation_inference_time", []).append(time.time() - start_time)
                        predictions["alignment_computation_inference_time"] = context["alignment_computation_inference_time"]

                if context is None:
                    predictions["pose_enc"] = [pose_enc_list[-1]]  # pose encoding of the last iteration
                    predictions["pose_enc_list"] = [pose_enc_list]
                else:

                    if mergeResults:
                        #pose_enc_list = merge_results(context_pose_enc_list, pose_enc_list, self.num_overlap)
                        context.setdefault("pose_enc", []).append(pose_enc_list[-1])
                        predictions["pose_enc"] = context["pose_enc"]
                        context.setdefault("pose_enc_list", []).append(pose_enc_list)
                        predictions["pose_enc_list"] = context["pose_enc_list"]
                    else:
                        predictions["pose_enc"] = [pose_enc_list[-1]]  # pose encoding of the last iteration
                        predictions["pose_enc_list"] = [pose_enc_list]


            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    filtered_aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )

                if gt_poses is not None:
                    #apply gt alignment scale
                    for b in range(B):
                        depth[b,...] *= batch_scales[b]

                if context is None:
                    predictions["depth"] = [depth]
                    predictions["depth_conf"] = [depth_conf]
                else:

                    #no need to align depth maps, since they are already aligned by the alignment head

                    if mergeResults:
                        context.setdefault("depth", []).append(depth)
                        predictions["depth"] = context["depth"]
                        context.setdefault("depth_conf", []).append(depth_conf)
                        predictions["depth_conf"] = context["depth_conf"]
                    else:
                        predictions["depth"] = [depth]
                        predictions["depth_conf"] = [depth_conf]

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    filtered_aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )

                #align point maps using last stage transform pose enc
                if self.camera_head is not None:
                    
                    if gt_poses is not None:
                        #apply gt alignment scale
                        for b in range(B):
                            pts3d[b,...] *= batch_scales[b]

                    if context is not None:
                        point_transform = closed_form_inverse_se3(mean_camera_transform.squeeze(1)).unsqueeze(1)  # (B, 1, 4, 4)
                        
                        #apply identity alignment
                        point_transform = point_transform @ point_identity_alignment.view(B,1,4,4)
                    else:
                        point_transform = point_identity_alignment.view(B,1,4,4)
                    
                    #convert points to homogeneous coordinates
                    pts3d_h = torch.cat([pts3d, torch.ones_like(pts3d[..., :1])], dim=-1) # (B, SC, H, W, 4)

                    pts3d_h = pts3d_h.view(B, -1, 4)  # (B, SC*H*W, 4)
                    point_transform = point_transform.expand(-1, pts3d_h.size(1), -1, -1)  # (B, SC*H*W, 4, 4)

                    pts3d_transformed = torch.matmul(point_transform, pts3d_h.unsqueeze(-1)).squeeze(-1)  # (B, SC*H*W, 4)

                    pts3d = pts3d_transformed[..., :3].view(B,S,H,W,3) # (B, SC, H, W, 3)


                if context is None:
                    predictions["world_points"] = [pts3d]
                    predictions["world_points_conf"] = [pts3d_conf]
                else:

                    #context_pts3d = context["world_points"][-1]
                    #context_pts3d_conf = context["world_points_conf"][-1]

                    if mergeResults:
                        context.setdefault("world_points", []).append(pts3d)
                        predictions["world_points"] = context["world_points"]
                        context.setdefault("world_points_conf", []).append(pts3d_conf)
                        predictions["world_points_conf"] = context["world_points_conf"]
                    else:
                        predictions["world_points"] = [pts3d]
                        predictions["world_points_conf"] = [pts3d_conf]
        
        return predictions
    

def align_poses(pose_enc, image_shape, num_overlap, context_pose_enc = None, gt_poses = None):
    B, S, _ = pose_enc.shape
    
    #only update last pose encoding, since only this is used during inference
    extr, intr = pose_encoding_to_extri_intri(pose_enc,image_size_hw=image_shape)
    extr = torch.nn.functional.pad(extr, (0,0,0,1,0,0,0,0), mode="constant")
    extr[:, :, 3, 3] = 1.

    #assure first pose is identity
    extr_identity_alignment = closed_form_inverse_se3(extr[:,0]) #B, 4, 4
    point_identity_alignment = extr[:,0]

    extr = extr @ extr_identity_alignment.view(B,1,4,4)

    #ground truth scale alignment
    if gt_poses is not None:
        #pad to 4x4
        gt_poses = torch.nn.functional.pad(gt_poses, (0,0,0,1,0,0,0,0), mode="constant")
        gt_poses[:, :, 3, 3] = 1.

        if S > 1:
            centering_transform = closed_form_inverse_se3(gt_poses[:,0]) #B, 4, 4

            gt_poses_firstFrameCentered = gt_poses @ centering_transform.view(B,1,4,4)

            #compute a gt scale alignment
            gt_positions = gt_poses_firstFrameCentered[...,:3,3].cpu().numpy()
            pred_positions = extr[...,:3,3].detach().cpu().numpy()

            batch_scales = []
            for b in range(B):
                batch_scale = scale_lse_solver(pred_positions[b],gt_positions[b])

                extr[b,:,:3,3] *= batch_scale

                batch_scales.append(batch_scale)
    
    if context_pose_enc is not None:
        if gt_poses is not None:
            #ground truth pose alignment
            mean_camera_transform = gt_poses[:,:1].to(extr) #B,1,4,4
        else:
            #grab last chunks pose encodings
            context_poses = pose_encoding_to_extri(context_pose_enc)
            context_overlapping = context_poses[:, -num_overlap:] # BxN_overlapx4x4

            overlapping = extr[:, :num_overlap]
            inv_overlapping = closed_form_inverse_se3(overlapping.reshape(B*num_overlap,4,4)).reshape(B,num_overlap,4,4)

            camera_transforms = inv_overlapping @ context_overlapping

            if num_overlap > 1:
                #TODO: opt supply other methods for computing alignment pose with multiple overlapping frames
                camera_transforms = extri_to_pose_encoding(camera_transforms)

                mean_camera_transform = averagePoseEncodings(camera_transforms) #torch.mean(camera_transforms, dim=1, keepdim=True)  # Bx1x7

                mean_camera_transform = pose_encoding_to_extri(mean_camera_transform) # Bx1x4x4
            else:
                mean_camera_transform = camera_transforms
    else:
        mean_camera_transform = torch.eye(4, device=extr.device, dtype=extr.dtype).view(1,1,4,4).expand(B,-1,-1,-1)
    
    adjusted_extr = extr @ mean_camera_transform #extri_to_pose_encoding(sc_poses @ mean_camera_transform)

    sc_adjusted_pose_enc = extri_intri_to_pose_encoding(adjusted_extr, intr, image_size_hw=image_shape)

    return sc_adjusted_pose_enc, point_identity_alignment, mean_camera_transform, batch_scales if gt_poses is not None else None


def extri_to_pose_encoding(
    extrinsics
):
    # extrinsics: BxSx3x4
    R = extrinsics[:, :, :3, :3]  # BxSx3x3
    T = extrinsics[:, :, :3, 3]  # BxSx3

    quat = mat_to_quat(R)

    #normalize just to be sure
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    pose_encoding = torch.cat([T, quat], dim=-1).float()

    return pose_encoding


def pose_encoding_to_extri(
    pose_encoding
): 
    #pose enc: BxSx7
    T = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]
    R = quat_to_mat(quat)
    extrinsics = torch.cat([R, T[..., None]], dim=-1)
    extrinsics = torch.nn.functional.pad(extrinsics, (0,0,0,1,0,0,0,0), mode="constant")
    extrinsics[:,:, 3, 3] = 1.

    return extrinsics

def averagePoseEncodings(pose_encodings: torch.Tensor) -> torch.Tensor:
    """
    Average given pose encodings. Quaternions are averaged using using Markley's method (without uniform weights)
    
    Args:
        pose_encodings (torch.Tensor): shape (B, N, 7), batch of sets of pose_encodings.
    
    Returns:
        torch.Tensor: shape (B, 4), average pose encodings.
    """
    B, N, _ = pose_encodings.shape

    translations = pose_encodings[...,:3]
    quaternions = pose_encodings[..., 3:7]

    #directly average translations
    avg_translation = torch.mean(translations,dim=1, keepdim=True)

    # Normalize quaternions just in case
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    weights = (torch.ones(B, N, device=quaternions.device, dtype=quaternions.dtype) / N).unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)

    outer_products = quaternions.unsqueeze(-1) * quaternions.unsqueeze(-2) #(B, N, 4, 4)

    M = (weights * outer_products).sum(dim=1)  # (B, 4, 4)

    # Compute eigenvalues/vectors of M
    _, eigvecs = torch.linalg.eigh(M)  # (B, 4), (B, 4, 4)
    
    # Select eigenvector with largest eigenvalue
    max_eigvec = eigvecs[..., -1]  # (B, 4)

    # Normalize
    avg_quat = max_eigvec / max_eigvec.norm(dim=-1, keepdim=True)

    return torch.cat([avg_translation, avg_quat.unsqueeze(1)], dim=-1).float()

def merge_results(first_chunk, second_chunk, num_overlap = 0, mergeDim = 1):

    #TODO: check if it rather makes sense selecting the overlapping frames from first or second chunk
        #      ~ second chunk logically makes more sense since we get gradients to first chunk poses through alignment
        #      Another idea: output both chunk results and apply loss over both of them

    #sc_adjusted_pose_enc_list = sc_adjusted_pose_enc_list[:,:,self.num_overlap:]

    if isinstance(first_chunk, list) and isinstance(second_chunk, list):

        if num_overlap > 0:
            #first_chunk = [item[:,:-num_overlap] for item in first_chunk]
            second_chunk = [item[:,num_overlap:] for item in second_chunk]
    
        merged = [torch.cat((fc_item, sc_item), dim=mergeDim) for fc_item, sc_item in zip(first_chunk, second_chunk)]

        first_chunk.clear()
        second_chunk.clear()
    else:
        if num_overlap > 0:
            #merged = torch.cat((first_chunk[:,:-num_overlap], second_chunk), dim=mergeDim)
            #first_chunk = first_chunk[:,:-num_overlap]
            second_chunk = second_chunk[:,num_overlap:]
        
        merged = torch.cat((first_chunk, second_chunk), dim=mergeDim)

    del first_chunk
    del second_chunk

    return merged


"""
    P = len(context_pose_enc_list)
    for i in range(P):
        context_poses = pose_encoding_to_extri(context_pose_enc_list[i])
        context_overlapping = context_poses[:, -self.num_overlap:] # BxN_overlapx4x4

        extr, intr = pose_encoding_to_extri_intri(pose_enc_list[i],image_size_hw=images.shape[-2:])
        extr = torch.nn.functional.pad(extr, (0,0,0,1,0,0,0,0), mode="constant")
        extr[:,:, 3, 3] = 1.
        overlapping = extr[:, :self.num_overlap]
        inv_overlapping = closed_form_inverse_se3(overlapping.reshape(B*self.num_overlap,4,4)).reshape(B,self.num_overlap,4,4)
        
        camera_transforms = inv_overlapping @ context_overlapping

        if self.num_overlap > 1:
            #TODO: opt supply other methods for computing alignment pose with multiple overlapping frames
            camera_transforms = extri_to_pose_encoding(camera_transforms)

            mean_camera_transform = averagePoseEncodings(camera_transforms) #torch.mean(camera_transforms, dim=1, keepdim=True)  # Bx1x7

            mean_camera_transform = pose_encoding_to_extri(mean_camera_transform) # Bx1x4x4
        else:
            mean_camera_transform = camera_transforms
        
        adjusted_extr = extr @ mean_camera_transform #extri_to_pose_encoding(sc_poses @ mean_camera_transform)

        sc_adjusted_pose_enc = extri_intri_to_pose_encoding(adjusted_extr, intr, image_size_hw=images.shape[-2:])
        pose_enc_list[i] = sc_adjusted_pose_enc
    """