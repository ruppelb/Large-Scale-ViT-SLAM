# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import torch
import torch.nn as nn
from typing import Tuple, Optional
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.vggt.models.aggregator import Aggregator
from vggt.vggt.heads.camera_head import CameraHead
from vggt.vggt.heads.dpt_head import DPTHead
from vggt.vggt.heads.track_head import TrackHead

from vggt.vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri

from aligned_vggt.utils.alignment import apply_sim3_alignment_on_w2c, apply_sim3_alignment_on_point_maps


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

    def set_config(self, cfg: dict):
        #self.num_overlap = cfg.num_overlap
        self.camera_head = self.camera_head if cfg.enable_camera else None
        self.point_head = self.point_head if cfg.enable_point else None
        self.depth_head = self.depth_head if cfg.enable_depth else None
        self.track_head = self.track_head if cfg.enable_track else None

    def forward(self, images: torch.Tensor, num_overlap, context : dict = None, gt_poses : torch.Tensor =None):
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

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    filtered_aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                
                if not self.training:
                    start_time = time.time()

                if context is not None:
                    #grab last chunks pose encodings
                    context_overlapping_point_maps = context["world_points"][-1][:, -num_overlap:] # BxN_overlapxHxWx3
                    context_overlapping_point_confidences = context["world_points_conf"][-1][:, -num_overlap:] # BxN_overlapxHxWx3

                    overlapping_point_maps = pts3d[:, :num_overlap]
                    overlapping_point_confidences = pts3d_conf[:, :num_overlap]

                    batch_transforms = []
                    batch_scales = []
                    for b in range(B):
                        
                        r,t,s = irls_sim3_umeyama(overlapping_point_maps[b],context_overlapping_point_maps[b],overlapping_point_confidences[b],context_overlapping_point_confidences[b])

                        pose = torch.nn.functional.pad(r,(0,1,0,1),mode="constant").to(pts3d)
                        pose[:3, 3] = t
                        pose[3, 3] = 1.
                        
                        batch_transforms.append(pose)
                        batch_scales.append(s)

                    alignment_transform = torch.stack(batch_transforms) # Bx4x4
                    batch_scales = torch.tensor(batch_scales).to(pts3d)
                else:
                    alignment_transform = torch.eye(4, device=images.device, dtype=images.dtype).view(1,4,4).expand(B,-1,-1)
                    batch_scales = torch.ones(B,device=pts3d.device,dtype=pts3d.dtype)

                if not self.training:
                    if context is None:
                        predictions["alignment_computation_inference_time"] = [time.time() - start_time]
                    else:
                        context.setdefault("alignment_computation_inference_time", []).append(time.time() - start_time)
                        predictions["alignment_computation_inference_time"] = context["alignment_computation_inference_time"]

                pts3d_final = apply_sim3_alignment_on_point_maps(pts3d,alignment_transform,batch_scales)

                if context is None:
                    predictions["world_points"] = [pts3d_final]
                    predictions["world_points_conf"] = [pts3d_conf]
                else:
                    context.setdefault("world_points", []).append(pts3d_final)
                    predictions["world_points"] = context["world_points"]
                    context.setdefault("world_points_conf", []).append(pts3d_conf)
                    predictions["world_points_conf"] = context["world_points_conf"]


            if self.camera_head is not None:
                pose_enc_list = self.camera_head(filtered_aggregated_tokens_list)

                #apply alignment inferred from points
                if self.point_head is not None:

                    #only update last pose encoding, since only this is used during inference
                    extr, intr = pose_encoding_to_extri_intri(pose_enc_list[-1],image_size_hw=images.shape[-2:])

                    #apply transform                    
                    adjusted_extr = apply_sim3_alignment_on_w2c(extr,alignment_transform,batch_scales)    

                    adjusted_pose_enc = extri_intri_to_pose_encoding(adjusted_extr, intr, image_size_hw=images.shape[-2:])
                    pose_enc_list[-1] = adjusted_pose_enc

                if context is None:
                    predictions["pose_enc"] = [pose_enc_list[-1]]  # pose encoding of the last iteration
                    predictions["pose_enc_list"] = [pose_enc_list]
                else:
                    context.setdefault("pose_enc", []).append(pose_enc_list[-1])
                    predictions["pose_enc"] = context["pose_enc"]
                    context.setdefault("pose_enc_list", []).append(pose_enc_list)
                    predictions["pose_enc_list"] = context["pose_enc_list"]


            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    filtered_aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )

                # apply scale of sim3 transform
                if self.point_head is not None:
                    for b in range(B):
                        depth[b,...] *= batch_scales[b]

                if context is None:
                    predictions["depth"] = [depth]
                    predictions["depth_conf"] = [depth_conf]
                else:
                    #no need to align depth maps, since they are already aligned by the alignment head
                    context.setdefault("depth", []).append(depth)
                    predictions["depth"] = context["depth"]
                    context.setdefault("depth_conf", []).append(depth_conf)
                    predictions["depth_conf"] = context["depth_conf"]

        return predictions

def weighted_umeyama_sim3(
    src: torch.Tensor,
    dst: torch.Tensor,
    weights: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    """
    Weighted Umeyama for SIM(3):
      minimize sum_i w_i || s R x_i + t - y_i ||^2
    Returns (s, R, t)
    src, dst: (M,3)
    weights: (M,)
    """

    assert src.ndim == 2 and src.shape[1] == 3
    assert dst.shape == src.shape

    M = src.shape[0]
    w = weights.view(M, 1)
    Wsum = weights.sum()

    if Wsum < 1e-6:
        raise ValueError("Total weight too small for meaningful estimation")

    # weighted centroids
    mu_x = (w * src).sum(dim=0) / Wsum   # source centroid
    mu_y = (w * dst).sum(dim=0) / Wsum   # destination centroid

    # centered
    x_c = src - mu_x
    y_c = dst - mu_y

    # weighted covariance
    Sigma = (w * y_c).T @ x_c  # 3x3
    Sigma = Sigma / Wsum

    U, S, Vh = torch.linalg.svd(Sigma, full_matrices=True)
    V = Vh.T

    # construct S_diag and compute R
    det_term = torch.det(U @ V.T)
    S_diag = torch.diag(torch.tensor([1.0, 1.0, torch.sign(det_term).item()], device=src.device, dtype=src.dtype))
    r = U @ S_diag @ V.T

    # weighted source variance
    var_x = (weights * (x_c**2).sum(dim=1)).sum() / Wsum
     
    # Umeyama scale: s = trace(S_diag * diag(Svals)) / var_x
    # diag(S) * S_diag elementwise
    c = (S * S_diag).sum() / var_x

    # translation
    t = mu_y - c * (r @ mu_x)

    return r, t, c

def irls_sim3_umeyama(
    src: torch.Tensor,
    dst: torch.Tensor,
    conf_src: Optional[torch.Tensor],
    conf_dst: Optional[torch.Tensor],
    conf_threshold_factor: float = 0.5,   # fraction of median combined confidence
    delta: float = 0.1,
    max_iters: int = 20,
    tol: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    IRLS-based SIM(3) estimation using weighted Umeyama at each iteration.

    Inputs:
      src, dst: (N,H,W,3) corresponding points
      conf_src, conf_dst: (N,H,W) per-point confidences.
      conf_threshold_factor: remove pairs with combined_conf < factor * median(combined_conf)
      delta: parameter for huber function (delta or c)
      max_iters, tol: IRLS control
      device, dtype: optional
    Returns:
    """

    #assume same number of point maps
    assert src.shape[0] == dst.shape[0]

    #convert point maps to two sets of points
    src = src.reshape(-1,3)
    dst = dst.reshape(-1,3)
    conf_src = conf_src.reshape(-1)
    conf_dst = conf_dst.reshape(-1)

    # combine per-point confidences
    combined = torch.sqrt(conf_src * conf_dst)

    # adaptive threshold relative to median
    conf_thresh = conf_threshold_factor * torch.median(combined)
    mask = combined >= conf_thresh

    #print(f"[IRLS] keeping {int(mask.sum().item())}/{N*H*W} correspondences (threshold = {conf_thresh.item():.6g})")

    src = src[mask]
    dst = dst[mask]
    combined = combined[mask]

    def huber_weights(r: torch.Tensor, delta: float):
        mask = r <= delta
        return torch.where(mask, torch.ones_like(r), delta / r.clamp_min(1e-12))
    
    # Initialize weights (start with combined confidences)
    weights = combined.clone()

    # initial guess from one Umeyama call
    R, t, s = weighted_umeyama_sim3(src, dst, weights)

    last_R = R.clone()
    last_t = t.clone()
    last_s = s.clone()

    for it in range(max_iters):
        # compute transformed points and residuals
        transformed = s * (src @ R.T) + t  # (M,3)
        residuals = torch.linalg.norm(transformed - dst, dim=1)  # (M,)

        # robust multiplicative weights
        robust_w = huber_weights(residuals, delta)
        new_weights = combined * robust_w

        #normalize weights
        #new_weights /= (torch.sum(new_weights) + 1e-12)

        # solve weighted Umeyama with new_weights
        R, t, s = weighted_umeyama_sim3(src, dst, new_weights)

        # convergence checks (norms)
        dR = torch.norm(R - last_R)
        dt = torch.norm(t - last_t)
        ds = torch.abs(s - last_s)
        
        mean_res = residuals.mean().item()
        wmin, wmean, wmax = new_weights.min().item(), new_weights.mean().item(), new_weights.max().item()
        #print(f"Iter {it:2d}: mean_res={mean_res:.6g}, ds={ds:.3e}, dR={dR:.3e}, dt={dt:.3e}, weights(min/mean/max)=({wmin:.3g},{wmean:.3g},{wmax:.3g})")
        
        last_R = R.clone()
        last_t = t.clone()
        last_s = s.clone()

        weights = new_weights

        if dR < tol and dt < tol and ds < tol:
            break

    # final diagnostics
    transformed = s * (src @ R.T) + t
    final_residuals = torch.linalg.norm(transformed - dst, dim=1)

    #print(f"iters {it}, mean_res={mean_res:.6g}, ds={ds:.3e}, dR={dR:.3e}, dt={dt:.3e}, weights(min/mean/max)=({wmin:.3g},{wmean:.3g},{wmax:.3g})")
    return R, t, s