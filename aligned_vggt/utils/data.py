import torch
import random
import logging
from typing import Optional, Tuple

from vggt.vggt.utils.rotation import quat_to_mat, mat_to_quat
from vggt.vggt.utils.geometry import closed_form_inverse_se3
from aligned_vggt.utils.alignment import *
from vggt.training.train_utils.general import check_and_fix_inf_nan


def extri_to_pose_encoding(extrinsics: torch.Tensor) -> torch.Tensor:
    """
    Convert extrinsics to pose encoding (translation + quaternion).
    Args:
        extrinsics (torch.Tensor): shape (B, S, 3, 4)
    Returns:
        pose_encoding (torch.Tensor): shape (B, S, 7)
    """
    R = extrinsics[:, :, :3, :3]
    T = extrinsics[:, :, :3, 3]

    quat = mat_to_quat(R)

    # normalize to get valid rotation quaternion
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    pose_encoding = torch.cat([T, quat], dim=-1).float()

    return pose_encoding


def pose_encoding_to_extri(pose_encoding: torch.Tensor) -> torch.Tensor:
    """
    Convert pose encoding (translation + quaternion) to extrinsics.
    Args:
        pose_encoding (torch.Tensor): shape (B, S, 7)
    Returns:
        extrinsics (torch.Tensor): shape (B, S, 3, 4)
    """
    T = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]

    # normalize to get valid rotation quaternion
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    R = quat_to_mat(quat)
    extrinsics = torch.cat([R, T[..., None]], dim=-1)
    extrinsics = torch.nn.functional.pad(extrinsics, (0,0,0,1,0,0,0,0), mode="constant")
    extrinsics[:,:, 3, 3] = 1.

    return extrinsics

def convertDictListsToTensors(chunked_dict: dict, overlap: int, out_dict: dict = None) -> None:
    """
    Convert lists in dict to tensors by concatenating along dimension 1.
    If out_dict is provided, store results there, otherwise modify chunked_dict in place.
    Args:
        chunked_dict (dict): dictionary with lists of tensors to convert.
        overlap (int): number of overlapping frames to omit from each chunk except the first.
        out_dict (dict, optional): dictionary to store results. If None, modify chunked_dict in place.
    Returns:
        None (results are stored in out_dict or chunked_dict).
    """

    if out_dict == None:
        out_dict = chunked_dict

    keys_to_merge = ["pose_enc", "pose_enc_list", "world_points", "world_points_conf", "depth", "depth_conf", "extrinsics", "intrinsics", "scales", "cam_points", "depths", "point_masks", "images", "ids"]

    for key in chunked_dict.keys():
        if key in keys_to_merge:
            if isinstance(chunked_dict[key][0],list):
                # pose encoding list
                if overlap > 0:
                    # omit overlapping frames
                    for i in range(1,len(chunked_dict[key])):
                        chunked_dict[key][i] = [item[:,overlap:] for item in chunked_dict[key][i]]

                out_dict[key] = [torch.cat(tensor_tuple, dim=1) for tensor_tuple in zip(*chunked_dict[key])]
            else:
                if overlap > 0:
                    # omit overlapping frames
                    for i in range(1,len(chunked_dict[key])):
                        chunked_dict[key][i] = chunked_dict[key][i][:,overlap:]

                out_dict[key] = torch.cat(chunked_dict[key],dim=1)

def moveDictListItemToCPU(chunked_dict: dict, itemIndex: int) -> None:
    """
    Move a specific item from a list in chunked_dict to CPU.
    
    Args:
        chunked_dict (dict): Dictionary containing lists of tensors.
        itemIndex (int): Index of the item in the list to move to CPU.
    """

    for key in chunked_dict.keys():
        if isinstance(chunked_dict[key],list):
            if len(chunked_dict[key]) >= (abs(itemIndex) if itemIndex < 0 else itemIndex+1):
                if isinstance(chunked_dict[key][0],list):
                    #pose encoding list
                    chunked_dict[key][itemIndex] = [(item.cpu() if isinstance(item, torch.Tensor) else item) for item in chunked_dict[key][itemIndex]]
                else:
                    if isinstance(chunked_dict[key][itemIndex], torch.Tensor):
                        chunked_dict[key][itemIndex] = chunked_dict[key][itemIndex].cpu()

def alignAndConvertOutputs(predictions: dict, batch: dict, chunked_batch: dict, alignment_type: str, seq_width: int, overlap: int) -> None:
    """
    Align and convert outputs.

    Args:
        predictions (dict): Predictions dictionary.
        batch (dict): Original batch dictionary.
        chunked_batch (dict): Chunked batch dictionary.
        alignment_type (str): Type of alignment to perform.
        seq_width (int): Sequence width.
        overlap (int): Overlap size.
        
    Returns:
        None: Results are stored in predictions and batch.
    """
    if alignment_type == "per_chunk_scale_from_poses":
        # for this we need chunked outputs
        per_chunk_scale_alignment_from_poses(predictions,chunked_batch)
        convertDictListsToTensors(chunked_batch,overlap,batch)
        convertDictListsToTensors(predictions,overlap)
    else:
        convertDictListsToTensors(chunked_batch,overlap,batch)
        convertDictListsToTensors(predictions,overlap)
        if alignment_type == "scale_from_fc_poses":
            scale_alignment_from_poses(predictions,batch,seq_width)
        elif alignment_type == "scale_from_poses":
            scale_alignment_from_poses(predictions,batch)
        elif alignment_type == "per_frame_scale_from_poses":
            per_frame_scale_alignment_from_poses(predictions,batch)
        elif alignment_type == "scale_from_depths":
            if "depth" not in predictions:
                raise ValueError("scale_from_depths alignment requires depth head to be enabled.")
            
            scale_align_from_depths(predictions,batch)
        elif alignment_type == "sim3_from_poses":
            umeyama_alignment_from_poses(predictions, batch, seq_width)
        elif alignment_type == "sim3_from_points":
            
            if "world_points" not in predictions:
                raise ValueError("sim3_from_points alignment requires point head to be enabled.")
            
            batch_transforms, batch_scales = umeyama_alignment_from_points(predictions["world_points"][:,:seq_width], predictions["world_points_conf"][:,:seq_width], batch["world_points"][:,:seq_width], batch["point_masks"][:,:seq_width], confidence_threshold=50.0) #90
            apply_sim3_alignment_on_dict(predictions, batch["images"].shape[-2:], batch_transforms, batch_scales)
        else:
            # no alignment
            pass

def generate_chunks(num_frames: int, mode: str, seq_width: int, overlap: int) -> list:
    """
    Generate chunk indices for a sequence.
    
    Args:
        num_frames (int): Total number of frames in the sequence.
        mode (str): Chunking mode ("chunk_gt", "chunk_overlap", "all", "two_chunks").
        seq_width (int): Width of each chunk.
        overlap (int): Overlap size between chunks (only for "chunk_overlap" mode).
    Returns:
        List[List[int]]: List of chunk indices.
    """
    indices = []
    if mode == "chunk_gt":

        # divide sequence in non-overlapping chunks of width seq_width
        for i in range(0, num_frames - seq_width + 1, seq_width):
            indices.append(list(range(i,i+seq_width)))

        # check if all images are at least within one sequence  
        if len(indices) * seq_width < num_frames:
            indices.append(list(range(len(indices) * seq_width, num_frames)))

    elif mode == "chunk_overlap":

        if num_frames < seq_width:
            indices.append(list(range(num_frames)))
        else:
            # divide sequence in overlapping chunks of width seq_width
            for i in range(0, num_frames - seq_width + 1, seq_width - overlap):
                indices.append(list(range(i,i+seq_width)))
                
            # check if all images are at least within one sequence  
            if len(indices) * (seq_width - overlap) < num_frames - overlap:
                # create a smaller chunk with the last images
                indices.append(list(range(len(indices) * (seq_width - overlap), num_frames)))
    elif mode == "all":
        indices = [list(range(num_frames))]
    elif mode == "two_chunks":
        # sample two non-overlapping chunks regardless of seq_width
        if num_frames < 2:
            raise ValueError("Number of frames must be at least 2 for two_chunks mode.")
        elif num_frames == 2:
            indices = [[0,1]]
        else:
            all_indices = list(range(num_frames))
            first_chunk_size = random.randint(1, num_frames - 1)
            first_chunk = random.sample(all_indices, first_chunk_size)
            second_chunk = [idx for idx in all_indices if idx not in first_chunk]
            indices = [first_chunk, second_chunk]
    else:
        raise ValueError(f"Unknown sequence generation mode: {mode}")
    return indices

def chunk_batch(batch: dict, indices: list) -> dict:
    """
    Chunk a batch of data according to the provided indices.
    
    Args:
        batch (dict): A dictionary containing batch data with tensors of shape [B, N, ...].
        indices (List[List[int]]): A list of lists, where each sublist contains the indices for a chunk.
    Returns:
        dict: A dictionary containing chunked batch data.
    """

    chunked_batch = {}
    for chunk_ids in indices:
        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor):
                chunked_batch.setdefault(key,[]).append(batch[key][:,chunk_ids])
    return chunked_batch

# Taken from VGGT, as imports do not work correctly else
def check_valid_tensor(input_tensor: Optional[torch.Tensor], name: str = "tensor") -> None:
    """
    Check if a tensor contains NaN or Inf values and log a warning if found.
    
    Args:
        input_tensor: The tensor to check
        name: Name of the tensor for logging purposes
    """
    if input_tensor is not None:
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            logging.warning(f"NaN or Inf found in tensor: {name}")


def normalize_camera_extrinsics_and_points_batch(
    extrinsics: torch.Tensor,
    cam_points: Optional[torch.Tensor] = None,
    world_points: Optional[torch.Tensor] = None,
    depths: Optional[torch.Tensor] = None,
    scale_by_points: bool = True,
    point_masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Normalize camera extrinsics and corresponding 3D points.
    
    This function transforms the coordinate system to be centered at the first camera
    and optionally scales the scene to have unit average distance.
    
    Args:
        extrinsics: Camera extrinsic matrices of shape (B, S, 3, 4)
        cam_points: 3D points in camera coordinates of shape (B, S, H, W, 3) or (*,3)
        world_points: 3D points in world coordinates of shape (B, S, H, W, 3) or (*,3)
        depths: Depth maps of shape (B, S, H, W)
        scale_by_points: Whether to normalize the scale based on point distances
        point_masks: Boolean masks for valid points of shape (B, S, H, W)
    
    Returns:
        Tuple containing:
        - Normalized camera extrinsics of shape (B, S, 3, 4)
        - Normalized camera points (same shape as input cam_points)
        - Normalized world points (same shape as input world_points)
        - Normalized depths (same shape as input depths)
    """
    # Validate inputs
    check_valid_tensor(extrinsics, "extrinsics")
    check_valid_tensor(cam_points, "cam_points")
    check_valid_tensor(world_points, "world_points")
    check_valid_tensor(depths, "depths")


    B, S, _, _ = extrinsics.shape
    device = extrinsics.device
    assert device == torch.device("cpu")


    # Convert extrinsics to homogeneous form: (B, N,4,4)
    extrinsics_homog = torch.cat(
        [
            extrinsics,
            torch.zeros((B, S, 1, 4), device=device),
        ],
        dim=-2,
    )
    extrinsics_homog[:, :, -1, -1] = 1.0

    # first_cam_extrinsic_inv, the inverse of the first camera's extrinsic matrix
    # which can be also viewed as the cam_to_world extrinsic matrix
    first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
    # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv)
    new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)


    if world_points is not None:
        # since we are transforming the world points to the first camera's coordinate system
        # we directly use the cam_from_world extrinsic matrix of the first camera
        # instead of using the inverse of the first camera's extrinsic matrix
        R = extrinsics[:, 0, :3, :3]
        t = extrinsics[:, 0, :3, 3]
        new_world_points = (world_points @ R.transpose(-1, -2).unsqueeze(1).unsqueeze(2)) + t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    else:
        new_world_points = None


    if scale_by_points:
        new_cam_points = cam_points.clone()
        new_depths = depths.clone()

        dist = new_world_points.norm(dim=-1)
        dist_sum = (dist * point_masks).sum(dim=[1,2,3])
        valid_count = point_masks.sum(dim=[1,2,3])
        avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)


        new_world_points = new_world_points / avg_scale.view(-1, 1, 1, 1, 1)
        new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / avg_scale.view(-1, 1, 1)
        if depths is not None:
            new_depths = new_depths / avg_scale.view(-1, 1, 1, 1)
        if cam_points is not None:
            new_cam_points = new_cam_points / avg_scale.view(-1, 1, 1, 1, 1)
    else:
        return new_extrinsics[:, :, :3], cam_points, new_world_points, depths

    new_extrinsics = new_extrinsics[:, :, :3] # 4x4 -> 3x4
    new_extrinsics = check_and_fix_inf_nan(new_extrinsics, "new_extrinsics", hard_max=None)
    new_cam_points = check_and_fix_inf_nan(new_cam_points, "new_cam_points", hard_max=None)
    new_world_points = check_and_fix_inf_nan(new_world_points, "new_world_points", hard_max=None)
    new_depths = check_and_fix_inf_nan(new_depths, "new_depths", hard_max=None)

    return new_extrinsics, new_cam_points, new_world_points, new_depths