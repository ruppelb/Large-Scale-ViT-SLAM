# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import torch
import numpy as np
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

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    filtered_aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                
                if not self.training:
                    start_time = time.time()

                if context is not None:
                    #grab last chunks pose encodings
                    context_overlapping_point_maps = context["world_points"][-1][:, -num_overlap:] #context["world_points_raw"][-1][:, -self.num_overlap:] # BxN_overlapxHxWx3
                    context_overlapping_point_confidences = context["world_points_conf"][-1][:, -num_overlap:] # BxN_overlapxHxWx3
                    #cumulative_alignment_transform = context["cumulative_transform"]
                    #cumulative_alignment_scales = context["cumulative_scales"]

                    overlapping_point_maps = pts3d[:, :num_overlap]
                    overlapping_point_confidences = pts3d_conf[:, :num_overlap]

                    batch_transforms = []
                    batch_scales = []
                    for b in range(B):
                        
                        r,t,s = irls_sim3_umeyama(overlapping_point_maps[b],context_overlapping_point_maps[b],overlapping_point_confidences[b],context_overlapping_point_confidences[b])

                        pose = torch.nn.functional.pad(r,(0,1,0,1),mode="constant").to(pts3d)
                        pose[:3, 3] = t
                        pose[3, 3] = 1.
                        
                        """
                        conf_threshold = min(np.median(overlapping_point_confidences[b].cpu().detach().numpy()), np.median(context_overlapping_point_confidences[b].cpu().detach().numpy())) #* 0.9
                        s, r, t = weighted_align_point_maps(context_overlapping_point_maps[b].cpu().detach().numpy(),context_overlapping_point_confidences[b].cpu().detach().numpy(),overlapping_point_maps[b].cpu().detach().numpy(),overlapping_point_confidences[b].cpu().detach().numpy(),conf_threshold)

                        # Convert to 4x4 matrices
                        pose = np.pad(r, ((0, 1), (0, 1)), mode="constant")
                        pose[:3, 3] = t
                        pose[3, 3] = 1.

                        pose = torch.from_numpy(pose).to(pts3d)

                        
                        #apply cumulative transform
                        s = cumulative_alignment_scales[b] * s

                        pose[:3,3] *= cumulative_alignment_scales[b]
                        pose = cumulative_alignment_transform[b] @ pose
                        """

                        batch_transforms.append(pose)
                        batch_scales.append(s)

                    alignment_transform = torch.stack(batch_transforms) #.to(pts3d) .unsqueeze(1) # Bx4x4
                    batch_scales = torch.tensor(batch_scales).to(pts3d)
                else:
                    alignment_transform = torch.eye(4, device=images.device, dtype=images.dtype).view(1,4,4).expand(B,-1,-1) #.view(1,1,4,4).expand(B,-1,-1,-1)
                    batch_scales = torch.ones(B,device=pts3d.device,dtype=pts3d.dtype)

                if not self.training:
                    if context is None:
                        predictions["alignment_computation_inference_time"] = [time.time() - start_time]
                    else:
                        context.setdefault("alignment_computation_inference_time", []).append(time.time() - start_time)
                        predictions["alignment_computation_inference_time"] = context["alignment_computation_inference_time"]

                pts3d_final = apply_sim3_alignment_on_point_maps(pts3d,alignment_transform,batch_scales)

                #predictions["cumulative_transform"] = alignment_transform
                #predictions["cumulative_scales"] = batch_scales
                if context is None:
                    predictions["world_points"] = [pts3d_final]
                    predictions["world_points_conf"] = [pts3d_conf]
                    #predictions["world_points_raw"] = [pts3d]
                else:

                    #context_pts3d = context["world_points"][-1]
                    #context_pts3d_conf = context["world_points_conf"][-1]

                    if mergeResults:
                        context.setdefault("world_points", []).append(pts3d_final)
                        predictions["world_points"] = context["world_points"]
                        context.setdefault("world_points_conf", []).append(pts3d_conf)
                        predictions["world_points_conf"] = context["world_points_conf"]
                        #context.setdefault("world_points_raw", []).append(pts3d)
                        #predictions["world_points_raw"] = context["world_points_raw"]
                    else:
                        predictions["world_points"] = [pts3d_final]
                        predictions["world_points_conf"] = [pts3d_conf]
                        #predictions["world_points_raw"] = [pts3d]


            if self.camera_head is not None:
                pose_enc_list = self.camera_head(filtered_aggregated_tokens_list)

                #apply alignment inferred from points
                if self.point_head is not None:

                    #only update last pose encoding, since only this is used during inference
                    extr, intr = pose_encoding_to_extri_intri(pose_enc_list[-1],image_size_hw=images.shape[-2:])
                    #extr = torch.nn.functional.pad(extr, (0,0,0,1,0,0,0,0), mode="constant")
                    #extr[:, :, 3, 3] = 1.

                    #c2w = closed_form_inverse_se3(extr.reshape(B*S,4,4)).reshape(B,S,4,4)

                    # scale translations
                    #for b in range(B):
                    #    c2w[b,:,:3,3] *= batch_scales[b]

                    #apply transform
                    #alignment_camera_transform = closed_form_inverse_se3(alignment_point_transform.squeeze(1)).unsqueeze(1)  # (B, 1, 4, 4)
                    #adjusted_extr = extr @ alignment_camera_transform
                    
                    adjusted_extr = apply_sim3_alignment_on_w2c(extr,alignment_transform,batch_scales)    

                    adjusted_pose_enc = extri_intri_to_pose_encoding(adjusted_extr, intr, image_size_hw=images.shape[-2:])
                    pose_enc_list[-1] = adjusted_pose_enc

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

                # apply scale of sim3 transform
                if self.point_head is not None:
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
        
        return predictions
    


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

    N,H,W = src.shape[:3]

    #print(f"Before shape src: {src.shape}, dest: {dst.shape}, conf_src {conf_src.shape},  conf_dest {conf_dst.shape}")
    #print(f"Medians before: src {torch.median(conf_src)}, dst {torch.median(conf_dst)}")

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

    print(f"[IRLS] keeping {int(mask.sum().item())}/{N*H*W} correspondences (threshold = {conf_thresh.item():.6g})")

    src = src[mask]
    dst = dst[mask]
    combined = combined[mask]

    """
    # clamp large confidences
    if conf_clamp_max is not None:
        combined = torch.clamp(combined, max=conf_clamp_max)
    elif conf_clamp_percentile is not None:
        if not (0.0 < conf_clamp_percentile < 1.0):
            raise ValueError("conf_clamp_percentile must be in (0,1)")
        thresh_val = torch.quantile(combined, conf_clamp_percentile)
        combined = torch.clamp(combined, max=thresh_val)
    """

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

    print(f"iters {it}, mean_res={mean_res:.6g}, ds={ds:.3e}, dR={dR:.3e}, dt={dt:.3e}, weights(min/mean/max)=({wmin:.3g},{wmean:.3g},{wmax:.3g})")
    return R, t, s



# <---------   VGGT-Long alignment  --------->
def weighted_align_point_maps(point_map1, conf1, point_map2, conf2, conf_threshold):
    """ point_map2 -> point_map1"""
    b1, _, _, _ = point_map1.shape
    b2, _, _, _ = point_map2.shape
    b = min(b1, b2)
    
    aligned_points1 = []
    aligned_points2 = []
    confidence_weights = []

    for i in range(b):
        mask1 = conf1[i] > conf_threshold
        mask2 = conf2[i] > conf_threshold
        valid_mask = mask1 & mask2

        idx = np.where(valid_mask)
        if len(idx[0]) == 0:
            continue

        pts1 = point_map1[i][idx]
        pts2 = point_map2[i][idx]

        combined_conf = np.sqrt(conf1[i][idx] * conf2[i][idx])
        
        aligned_points1.append(pts1)
        aligned_points2.append(pts2)
        confidence_weights.append(combined_conf)

    if len(aligned_points1) == 0:
        raise ValueError("No matching point pairs were found!")

    all_pts1 = np.concatenate(aligned_points1, axis=0)
    all_pts2 = np.concatenate(aligned_points2, axis=0)
    all_weights = np.concatenate(confidence_weights, axis=0)

    print(f"The number of corresponding points matched: {all_pts1.shape[0]}")

    s, R, t = robust_weighted_estimate_sim3(all_pts2, 
                                            all_pts1, 
                                            all_weights,
                                            delta=0.1,
                                            max_iters=5,
                                            tol=1e-9)

    return s, R, t


def huber_loss(r, delta):
    abs_r = np.abs(r)
    return np.where(abs_r <= delta, 0.5 * r**2, delta * (abs_r - 0.5 * delta))

def robust_weighted_estimate_sim3(src, tgt, init_weights, delta=0.1, max_iters=20, tol=1e-9):
    """
    src:  (Nx3)
    tgt:  (Nx3)
    init_weights:  (N,)
    """
        
    s, R, t = weighted_estimate_sim3(src, tgt, init_weights)
    prev_error = float('inf')
    
    for iter in range(max_iters):

        transformed = s * (src @ R.T) + t
        residuals = np.linalg.norm(tgt - transformed, axis=1)  # (N,)
        print(f'Residuals: {np.mean(residuals)}')
        
        abs_res = np.abs(residuals)
        huber_weights = np.ones_like(residuals)
        large_res_mask = abs_res > delta
        huber_weights[large_res_mask] = delta / abs_res[large_res_mask]
        
        combined_weights = init_weights * huber_weights
        
        combined_weights /= (np.sum(combined_weights) + 1e-12)
        
        s_new, R_new, t_new = weighted_estimate_sim3(src, tgt, combined_weights)

        param_change = np.abs(s_new - s) + np.linalg.norm(t_new - t)
        rot_angle = np.arccos(min(1.0, max(-1.0, (np.trace(R_new @ R.T) - 1)/2)))
        current_error = np.sum(huber_loss(residuals, delta) * init_weights)
        
        if (param_change < tol and rot_angle < np.radians(0.1)) or \
           (abs(prev_error - current_error) < tol * prev_error):
            break

        s, R, t = s_new, R_new, t_new
        prev_error = current_error
    
    return s, R, t

def weighted_estimate_sim3(source_points, target_points, weights):
    """
    source_points:  (Nx3)
    target_points:  (Nx3)
    :weights:  (N,) [0,1]
    """
    total_weight = np.sum(weights)
    if total_weight < 1e-6:
        raise ValueError("Total weight too small for meaningful estimation")
    
    normalized_weights = weights / total_weight

    mu_src = np.sum(normalized_weights[:, None] * source_points, axis=0)
    mu_tgt = np.sum(normalized_weights[:, None] * target_points, axis=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    scale_src = np.sqrt(np.sum(normalized_weights * np.sum(src_centered**2, axis=1)))
    scale_tgt = np.sqrt(np.sum(normalized_weights * np.sum(tgt_centered**2, axis=1)))
    s = scale_tgt / scale_src

    weighted_src = (s * src_centered) * np.sqrt(normalized_weights)[:, None]
    weighted_tgt = tgt_centered * np.sqrt(normalized_weights)[:, None]
    
    H = weighted_src.T @ weighted_tgt

    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mu_tgt - s * R @ mu_src
    return s, R, t


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