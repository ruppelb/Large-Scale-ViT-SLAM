# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from dataclasses import dataclass
from vggt.vggt.utils.pose_enc import extri_intri_to_pose_encoding,pose_encoding_to_extri_intri
from vggt.vggt.utils.rotation import mat_to_quat
from vggt.training.train_utils.general import check_and_fix_inf_nan
from math import ceil, floor
from aligned_vggt.utils.geometry import compute_relative_poses

@dataclass(eq=False)
class MultitaskLoss(torch.nn.Module):
    """
    Multi-task loss module that combines different loss types for VGGT.
    
    Supports:
    - Camera loss
    - Depth loss 
    - Point loss
    - Tracking loss (not cleaned yet, dirty code is at the bottom of this file)
    """
    def __init__(self, perFrameReg=None, perChunkReg=None, depth = None, cameraPose = None, cameraPoseRel = None, **kwargs):
        super().__init__()
        # Loss configuration dictionaries for each task
        self.perFrameReg = perFrameReg
        self.perChunkReg = perChunkReg
        self.depth = depth
        self.cameraPose = cameraPose
        self.cameraPoseRel = cameraPoseRel

    def setupScheduling(self, total_steps):
        self.total_steps = total_steps

    def compute_warmup_weight(self, loss_type_dict, current_step):

        end_weight = loss_type_dict["weight"]

        #gather config
        if "warmup_percent" in loss_type_dict:
            warmup_steps = floor(self.total_steps * loss_type_dict["warmup_percent"])
        else:
            warmup_steps = 0

        #if no start step is specified, start at beginning
        if "warmup_start_percent" in loss_type_dict:
            start_step = floor(self.total_steps * loss_type_dict["warmup_start_percent"])
        else:
            start_step = 0.

        if "warmup_start_weight" in loss_type_dict:
            start_weight = loss_type_dict["warmup_start_weight"]
        else:
            start_weight = 0.

        #use exp as standard warmup
        if "warmup_type" in loss_type_dict:
            warmup_type = loss_type_dict["warmup_type"]
        else:
            warmup_type = "exp"

        #early outs
        if warmup_steps <= 0: return end_weight #0.0 no warmup
        if start_step > current_step: return 0.0 # we haven't reached the step from which this loss starts contributing
        if (start_step + warmup_steps) < current_step: return end_weight #1.0 fully warmed up

        delta_steps = current_step - start_step
        delta_weights = end_weight - start_weight

        #compute warmup factor by type
        if warmup_type == "exp":
            warmup_factor = (min(1.0, (float(delta_steps) / float(warmup_steps)))) ** self.weight_warmup_exp
        elif warmup_type == "linear":
            warmup_factor = (min(1.0, (float(delta_steps) / float(warmup_steps))))
        else:
            raise ValueError("Invalid learning rate warmup type")
        
        #print(f"Final weight: {start_weight + delta_weights * warmup_factor}")
        
        return start_weight + delta_weights * warmup_factor


    def forward(self, predictions, batch, current_step) -> torch.Tensor:
        """
        Compute the total multi-task loss.
        
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks
            
        Returns:
            Dict containing individual losses and total objective
        """

        total_loss = 0
        loss_dict = {}
        
        if ("per_frame_scales" in predictions or "per_frame_pose_enc_list" in predictions or "per_frame_pose_enc" in predictions) and self.perFrameReg is not None:
            per_frame_reg_loss_dict = per_frame_regularization_loss(predictions, **self.perFrameReg)
            per_frame_reg_loss = per_frame_reg_loss_dict["loss_per_frame_reg"] * self.compute_warmup_weight(self.perFrameReg, current_step)
            total_loss = total_loss + per_frame_reg_loss
            loss_dict.update(per_frame_reg_loss_dict)
        
        if ("per_chunk_scales" in predictions or "per_chunk_pose_enc_list" in predictions or "per_chunk_pose_enc" in predictions) and self.perChunkReg is not None:
            per_chunk_reg_loss_dict = per_chunk_regularization_loss(predictions, **self.perChunkReg)            
            per_chunk_reg_loss = per_chunk_reg_loss_dict["loss_per_chunk_reg"] * self.compute_warmup_weight(self.perChunkReg, current_step)
            total_loss = total_loss + per_chunk_reg_loss
            loss_dict.update(per_chunk_reg_loss_dict)
        
        if "depth" in predictions and self.depth is not None:
            depth_scale_loss_dict = compute_depth_scale_loss(predictions, batch, **self.depth)
            depth_scale_loss = depth_scale_loss_dict["loss_depth_scale"] * self.compute_warmup_weight(self.depth, current_step)
            total_loss = total_loss + depth_scale_loss
            loss_dict.update(depth_scale_loss_dict)

        if "pose_enc_list" in predictions and self.cameraPose is not None:
            camera_pose_loss_dict = compute_camera_pose_loss(predictions, batch, **self.cameraPose)        
            camera_pose_loss = camera_pose_loss_dict["loss_camera"] * self.compute_warmup_weight(self.cameraPose, current_step)
            total_loss = total_loss + camera_pose_loss
            loss_dict.update(camera_pose_loss_dict)

        if "pose_enc_list" in predictions and self.cameraPoseRel is not None:
            camera_pose_rel_dict = compute_relative_pose_loss(predictions, batch, **self.cameraPoseRel)
            camera_pose_rel_loss = camera_pose_rel_dict["loss_camera_rel"] * self.compute_warmup_weight(self.cameraPoseRel, current_step)   
            total_loss = total_loss + camera_pose_rel_loss
            loss_dict.update(camera_pose_rel_dict)

        loss_dict["objective"] = total_loss

        return loss_dict
    
def compute_camera_pose_loss(
    pred_dict,              # predictions dict, contains pose encodings
    batch_data,             # ground truth and mask batch dict
    loss_type="l1",         # "l1" or "l2" loss
    pose_encoding_type="absT_quaR_FoV",
    **kwargs):
    
    # List of predicted pose encodings per stage
    pred_pose_encodings = pred_dict['pose_enc_list']
    # Binary mask for valid points per frame (B, N, H, W)
    point_masks = batch_data['point_masks']
    # Only consider frames with enough valid points (>100)
    valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100

    # Get ground truth camera extrinsics and intrinsics
    gt_extrinsics = batch_data['extrinsics']
    gt_intrinsics = batch_data['intrinsics']
    image_hw = batch_data['images'].shape[-2:]

    # Encode ground truth pose to match predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )

    pred_pose_encoding = pred_pose_encodings[-1]

    if valid_frame_mask.sum() == 0:
        # If no valid frames, set losses to zero to avoid gradient issues
        loss_T = (pred_pose_encoding * 0).mean()
        loss_R = (pred_pose_encoding * 0).mean()
    else:
        if loss_type == "l1":
            # Translation: first 3 dims; Rotation: next 4 (quaternion);
            loss_T = (pred_pose_encoding[..., :3] - gt_pose_encoding[..., :3]).abs()
            loss_R = (pred_pose_encoding[..., 3:7] - gt_pose_encoding[..., 3:7]).abs()
        elif loss_type == "l2":
            # L2 norm for each component
            loss_T = (pred_pose_encoding[..., :3] - gt_pose_encoding[..., :3]).norm(dim=-1, keepdim=True)
            loss_R = (pred_pose_encoding[..., 3:7] - gt_pose_encoding[..., 3:7]).norm(dim=-1)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Check/fix numerical issues (nan/inf) for each loss component
        loss_T = check_and_fix_inf_nan(loss_T, "loss_T_pose")
        loss_R = check_and_fix_inf_nan(loss_R, "loss_R_pose")

        # Clamp outlier translation loss to prevent instability, then average
        loss_T = loss_T.clamp(max=100).mean()
        loss_R = loss_R.mean()

    # Compute total weighted camera loss
    total_camera_loss = (loss_T + loss_R)

    # Return loss dictionary with individual components
    return {
        "loss_camera": total_camera_loss,
        "loss_T": loss_T,
        "loss_R": loss_R
    }

def compute_depth_scale_loss(predictions, batch, valid_range = -1, **kwargs):
    """
    """
    pred_depth = predictions['depth'] #B, S, H, W, 1
    pred_depth_conf = predictions['depth_conf'] #B, S, H, W

    gt_depth = batch['depths'] #B, S, H, W
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth", hard_max=None)
    gt_depth = gt_depth[..., None]              # (B, S, H, W, 1)
    gt_depth_mask = batch['point_masks'].clone()   # 3D points derived from depth map, so we use the same mask

    if gt_depth_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_depth).mean()
        loss_dict = {f"loss_depth_scale": dummy_loss}
        return loss_dict
    
    #scale confidences to be between 0 and 1
    pred_depth_conf = pred_depth_conf / pred_depth_conf.amax(dim=(2,3), keepdim=True).clamp_min(1e-8)
    
    # Compute log L1 distance between predicted and ground truth depths in log space
    loss_reg = (torch.log(pred_depth[gt_depth_mask].clamp_min(1e-8)) - torch.log(gt_depth[gt_depth_mask].clamp_min(1e-8))).abs().squeeze(-1) # log loss
    #loss_reg = (pred_depth[gt_depth_mask] - gt_depth[gt_depth_mask]).abs().squeeze(-1)
    loss_depth_scale = loss_reg * pred_depth_conf[gt_depth_mask]
    
    #print(f"Mean per-frame log scale: {loss_depth_scale.mean().item()}, max: {loss_depth_scale.max().item()}, min: {loss_depth_scale.min().item()}")
    loss_depth_scale = check_and_fix_inf_nan(loss_depth_scale, "loss_depth_scale")

    # Process regular regression loss
    if loss_depth_scale.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_depth_scale = filter_by_quantile(loss_depth_scale, valid_range)

        loss_depth_scale = check_and_fix_inf_nan(loss_depth_scale, f"loss_depth_scale")
        loss_depth_scale = loss_depth_scale.mean()
    else:
        loss_depth_scale = (0.0 * pred_depth).mean()

    return {"loss_depth_scale": loss_depth_scale}

def per_frame_regularization_loss(
    pred_dict,
    gamma=0.6,              # temporal decay weight for multi-stage training
    **kwargs
):

    loss = 0.0

    if "per_frame_scales" in pred_dict:
        frames_scales = torch.cat(pred_dict["per_frame_scales"],dim=1) #B, S, 1
        loss = torch.mean(torch.log(frames_scales.clamp_min(1e-6))**2)

    if "per_frame_pose_enc" in pred_dict:
        frames_pose_enc_list = pred_dict["per_frame_pose_enc"].view(-1,7) #tensor of B,num_chunks*chunk_width, 7

        translation = frames_pose_enc_list[...,:3]
        rotation = frames_pose_enc_list[...,3:7]

        #assure quats are normalized
        rotation = rotation / rotation.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        loss_t = translation.norm(dim=-1, keepdim=True)
        loss_r = (1-(rotation[..., -1]**2)).abs() #2.0 * torch.acos(torch.clamp(rotation_stage[..., -1].abs(), -1.0 + 1e-8, 1.0 - 1e-8)) #geodesic distance, quat scalar is last

        loss_t = check_and_fix_inf_nan(loss_t, "frame_reg_t")
        loss_r = check_and_fix_inf_nan(loss_r, "frame_reg_r")

        loss = (loss_t.clamp(max=100).mean() + loss_r.mean())

    if "per_frame_pose_enc_list" in pred_dict:
        
        frames_pose_enc_list = pred_dict["per_frame_pose_enc_list"] #list of B, S, 8
        # Number of prediction stages
        n_stages = len(frames_pose_enc_list)

        # Compute loss for each prediction stage with temporal weighting
        total_loss_t = total_loss_r = total_loss_s = 0
        for stage_idx in range(n_stages):
            # Later stages get higher weight (gamma^0 = 1.0 for final stage)
            stage_weight = gamma ** (n_stages - stage_idx - 1)
            pred_pose_stage = frames_pose_enc_list[stage_idx]

            translation_stage = pred_pose_stage[...,:3]
            rotation_stage = pred_pose_stage[...,3:7]
            scale_stage = pred_pose_stage[...,7:8]

            #assure quats are normalized
            rotation_stage = rotation_stage / rotation_stage.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            loss_t_stage = translation_stage.norm(dim=-1, keepdim=True)
            loss_r_stage = (1-(rotation_stage[..., -1]**2)).abs()  #2.0 * torch.acos(torch.clamp(rotation_stage[..., -1].abs(), -1.0 + 1e-8, 1.0 - 1e-8)) #geodesic distance, quat scalar is last
            loss_s_stage = torch.log(scale_stage.clamp_min(1e-6))**2

            loss_t_stage = check_and_fix_inf_nan(loss_t_stage, "frame_reg_t_stage")
            loss_r_stage = check_and_fix_inf_nan(loss_r_stage, "frame_reg_r_stage")
            loss_s_stage = check_and_fix_inf_nan(loss_s_stage, "frame_reg_s_stage")

            total_loss_t += (loss_t_stage.clamp(max=100).mean() * stage_weight)
            total_loss_r += (loss_r_stage.mean() * stage_weight)
            total_loss_s += (loss_s_stage.mean() * stage_weight)

        loss = (total_loss_t + total_loss_r + total_loss_s) / n_stages

    return {"loss_per_frame_reg": loss}


def per_chunk_regularization_loss(
    pred_dict,
    gamma=0.6,              # temporal decay weight for multi-stage training
    **kwargs
):
    loss = 0.0

    if "per_chunk_scales" in pred_dict:
        chunk_scales = torch.stack(pred_dict["per_chunk_scales"],dim=1) #B, 1
        loss = torch.mean(torch.log(chunk_scales.clamp_min(1e-6))**2)
    
    if "per_chunk_pose_enc" in pred_dict:
        chunk_pose_enc = pred_dict["per_chunk_pose_enc"] #tensor of B*num_chunks, 1 , 7 or 8

        translation = chunk_pose_enc[...,:3]
        rotation = chunk_pose_enc[...,3:7]

        #assure quats are normalized
        rotation = rotation / rotation.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        loss_t = translation.norm(dim=-1, keepdim=True)
        loss_r = (1-(rotation[..., -1]**2)).abs() #2.0 * torch.acos(torch.clamp(rotation_stage[..., -1].abs(), -1.0 + 1e-8, 1.0 - 1e-8)) #geodesic distance, quat scalar is last

        loss_s = 0.0
        if chunk_pose_enc.shape[-1] == 8:
            scale = chunk_pose_enc[...,7]
            loss_s = torch.log(scale.clamp_min(1e-6))**2

        loss_t = check_and_fix_inf_nan(loss_t, "chunk_reg_t")
        loss_r = check_and_fix_inf_nan(loss_r, "chunk_reg_r")
        loss_s = check_and_fix_inf_nan(loss_s,"chunk_reg_s")

        loss = (loss_t.clamp(max=100).mean() + loss_r.mean() + loss_s.mean())

    if "per_chunk_pose_enc_list" in pred_dict:
        
        chunk_pose_enc_list = pred_dict["per_chunk_pose_enc_list"] #list of B, 8
        # Number of prediction stages
        n_stages = len(chunk_pose_enc_list)

        # Compute loss for each prediction stage with temporal weighting
        total_loss_t = total_loss_r = 0.0
        for stage_idx in range(n_stages):
            # Later stages get higher weight (gamma^0 = 1.0 for final stage)
            stage_weight = gamma ** (n_stages - stage_idx - 1)
            pred_pose_stage = chunk_pose_enc_list[stage_idx]

            translation_stage = pred_pose_stage[...,:3]
            rotation_stage = pred_pose_stage[...,3:7]

            #assure quats are normalized
            rotation_stage = rotation_stage / rotation_stage.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            loss_t_stage = translation_stage.norm(dim=-1, keepdim=True)
            loss_r_stage = (1-(rotation_stage[..., -1]**2)).abs() #2.0 * torch.acos(torch.clamp(rotation_stage[..., -1].abs(), -1.0 + 1e-8, 1.0 - 1e-8)) #geodesic distance, quat scalar is last

            loss_t_stage = check_and_fix_inf_nan(loss_t_stage, "chunk_reg_t_stage")
            loss_r_stage = check_and_fix_inf_nan(loss_r_stage, "chunk_reg_r_stage")

            total_loss_t += (loss_t_stage.clamp(max=100).mean() * stage_weight)
            total_loss_r += (loss_r_stage.mean() * stage_weight)

        loss = (total_loss_t + total_loss_r) / n_stages
    
    return {"loss_per_chunk_reg": loss}

def compute_relative_pose_loss(
    pred_dict,              # predictions dict, contains pose encodings
    batch_data,             # ground truth and mask batch dict
    loss_type="l1",         # "l1" or "l2" loss
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=1.0,       # weight for translation loss
    weight_rot=1.0,         # weight for rotation loss
    scale_agnostic = False,
    **kwargs
):
    
    pred_pose_encodings = pred_dict['pose_enc_list'][-1]
    pred_extr, _ = pose_encoding_to_extri_intri(pred_pose_encodings,pose_encoding_type=pose_encoding_type,build_intrinsics=False)

    # Get ground truth camera extrinsics
    gt_extrinsics = batch_data['extrinsics']

    # Binary mask for valid points per frame (B, N, H, W)
    point_masks = batch_data['point_masks']
    # Only consider frames with enough valid points (>100)
    valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100

    # If no valid frames, set losses to zero to avoid gradient issues
    if valid_frame_mask.sum() == 0:
        return {"loss_camera_rel": (pred_pose_encodings * 0).mean(), 
                "loss_T_rel":(pred_pose_encodings * 0).mean(),
                "loss_R_rel":(pred_pose_encodings * 0).mean()}

    S = gt_extrinsics.shape[1]
    large_offset = torch.randint(int(S/2),S,(1,)).item()

    #compute pairwise relative poses
    gt_pair_rel_poses = compute_relative_poses(gt_extrinsics)
    pred_pair_rel_poses = compute_relative_poses(pred_extr)

    #compute larger offset relative poses (half of sequence dimension should be enough)
    gt_far_rel_poses = compute_relative_poses(gt_extrinsics, large_offset)
    pred_far_rel_poses = compute_relative_poses(pred_extr, large_offset)
    
    gt_rel_poses = torch.cat((gt_pair_rel_poses,gt_far_rel_poses),dim=1)
    pred_rel_poses = torch.cat((pred_pair_rel_poses,pred_far_rel_poses),dim=1)

    gt_rel_quat = mat_to_quat(gt_rel_poses[:, :, :3, :3])
    pred_rel_quat = mat_to_quat(pred_rel_poses[:, :, :3, :3])

    gt_rel_t = gt_rel_poses[:, :, :3, 3]
    pred_rel_t = pred_rel_poses[:, :, :3, 3]

    #normalize translations to be scale agnostic
    if scale_agnostic:
        gt_rel_t = gt_rel_t / gt_rel_t.norm(dim=-1,keepdim=True)
        pred_rel_t = pred_rel_t / pred_rel_t.norm(dim=-1,keepdim=True)

    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T = (pred_rel_t - gt_rel_t).abs()
        loss_R = (pred_rel_quat - gt_rel_quat).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_rel_t - gt_rel_t).norm(dim=-1, keepdim=True)
        loss_R = (pred_rel_quat - gt_rel_quat).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T_rel")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R_rel")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()

    # Return loss dictionary with individual components
    return {
        "loss_camera_rel": (weight_trans * loss_T) + (weight_rot * loss_R),
        "loss_T_rel": loss_T,
        "loss_R_rel": loss_R,
    }

def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filter loss tensor by keeping only values below a certain quantile threshold.
    
    This helps remove outliers that could destabilize training.
    
    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss
    
    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= min_elements:
        # Too few elements, just return as-is
        return loss_tensor

    # Randomly sample if tensor is too large to avoid memory issues
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values to prevent extreme outliers
    loss_tensor = loss_tensor.clamp(max=hard_max)

    # Compute quantile threshold
    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor

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

