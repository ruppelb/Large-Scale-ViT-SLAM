import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.vggt.models.aggregator import Aggregator
from vggt.vggt.heads.camera_head import CameraHead
from vggt.vggt.heads.dpt_head import DPTHead
from vggt.vggt.heads.track_head import TrackHead

from aligned_vggt.heads.alignment_head import AlignmentHead

from vggt.vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from vggt.vggt.utils.rotation import quat_to_mat, mat_to_quat
from vggt.vggt.utils.geometry import closed_form_inverse_se3

class FeatureAlignedVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True, num_memory_tokens = 8, temporal_attention = True, simple_decoder = False):
        
        super().__init__()

        self.embed_dim = embed_dim
        self.enable_memory = num_memory_tokens > 0

        self.intermediate_layer_indices = [4, 11, 17, 23]  # indices of intermediate layers
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1", intermediate_layer_idx=range(len(self.intermediate_layer_indices))) if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1", intermediate_layer_idx=range(len(self.intermediate_layer_indices))) if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

        self.alignment_head = AlignmentHead(in_dim=2*embed_dim,patch_size=patch_size,num_global_tokens= num_memory_tokens,temporal_attention=temporal_attention,simple_decoder=simple_decoder)

    def set_config(self, cfg: dict):
        #only called after we load the model from PyTorchModelHubMixin
        
        self.camera_head = self.camera_head if cfg.enable_camera else None
        self.point_head = self.point_head if cfg.enable_point else None
        self.depth_head = self.depth_head if cfg.enable_depth else None
        self.track_head = self.track_head if cfg.enable_track else None

        self.enable_memory = cfg.num_memory_tokens > 0

        self.alignment_head = AlignmentHead(in_dim=2*self.embed_dim,patch_size=cfg.patch_size,num_global_tokens = cfg.num_memory_tokens,temporal_attention=cfg.temporal_attention,simple_decoder=cfg.simple_decoder)

    def forward(self, images: torch.Tensor, num_overlap, context : dict = None, gt_poses : torch.Tensor =None):
        #context is a dict with lists or None for the first chunk

        B, S, C, H, W = images.shape

        predictions = {}

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        filtered_aggregated_tokens_list = [aggregated_tokens_list[i] for i in self.intermediate_layer_indices]
        aggregated_tokens_list.clear()
        del aggregated_tokens_list
        torch.cuda.empty_cache()

        #compute scale alignment based on last layer tokens
        context_overlap_tokens = None
        context_global_tokens = None
        if context is not None:
            context_overlap_tokens = context["overlap_tokens"]
            if self.enable_memory:
                context_global_tokens = context["global_tokens"][-1]

        #handle case that we get a chunk smaller than num overlap (only possible when we sample a single chunk during inference)
        overlap = num_overlap if S > num_overlap else S-1
        chunk_pose_enc, frame_pose_enc, global_tokens, overlap_tokens = self.alignment_head(filtered_aggregated_tokens_list[-1], (H, W), overlap, overlap_tokens=context_overlap_tokens, memory_tokens=context_global_tokens)

        #compute per-frame extrinsics
        chunk_alignment = pose_encoding_to_extri(chunk_pose_enc) #B,1,4,4
        chunk_scale = chunk_pose_enc[...,-1] #B,1
        
        frame_alignments = pose_encoding_to_extri(frame_pose_enc) #B,S-1,4,4
        frame_alignments = torch.matmul(frame_alignments, chunk_alignment)#torch.matmul(frame_alignments, chunk_alignment.detach())
        
        per_frame_extr = torch.cat([chunk_alignment,frame_alignments],dim=1) #B,S,4,4

        #decode poses, depth, points maps, and tracks for each chunk seperately
        with torch.amp.autocast("cuda", enabled=False): #with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(filtered_aggregated_tokens_list)

                #decode final vggt pose outputs
                extr, intr = pose_encoding_to_extri_intri(pose_enc_list[-1],image_size_hw=images.shape[-2:]) #B,S,3,4
                extr = torch.nn.functional.pad(extr, (0,0,0,1,0,0,0,0), mode="constant")
                extr[:,:, 3, 3] = 1.

                #assure first pose is identity
                extr_identity_alignment = closed_form_inverse_se3(extr[:,0]) #B, 4, 4
                if self.point_head is not None:
                    point_identity_alignment = extr[:,0].detach().clone()
                extr = extr @ extr_identity_alignment.view(B,1,4,4)

                #apply scale correction
                extr[:,:,:3,3] *= chunk_scale.view(B,1,1)

                #get initial chunk alignment estimate if context is given
                if context is not None:
                    if gt_poses is not None:
                        mean_camera_transform = gt_poses[:,:1].to(extr) #B,1,4,4
                    else:
                        context_overlapping = pose_encoding_to_extri(context["pose_enc_list"][-1][-1][:, -overlap:]) # BxN_overlapx4x4

                        inv_overlapping = closed_form_inverse_se3(extr[:, :overlap].reshape(B*overlap,4,4)).reshape(B,overlap,4,4)
                        
                        camera_transforms = inv_overlapping @ context_overlapping

                        if overlap > 1:
                            camera_transforms = extri_to_pose_encoding(camera_transforms)

                            mean_camera_transform = averagePoseEncodings(camera_transforms) #torch.mean(camera_transforms, dim=1, keepdim=True)  # Bx1x7

                            mean_camera_transform = pose_encoding_to_extri(mean_camera_transform) # Bx1x4x4
                        else:
                            mean_camera_transform = camera_transforms
                else:
                    mean_camera_transform = torch.eye(4, device=images.device, dtype=images.dtype).view(1,1,4,4).expand(B,-1,-1,-1)
                    
                per_frame_extr = torch.matmul(per_frame_extr, mean_camera_transform)

                #apply final pose alignment
                aligned_extr = torch.matmul(extr, per_frame_extr)
                
                aligned_pose_enc = extri_intri_to_pose_encoding(aligned_extr, intr, image_size_hw=images.shape[-2:])

                pose_enc_list[-1] = aligned_pose_enc

                #save dict keys
                predictions["overlap_tokens"] = overlap_tokens

                if context is None:
                    predictions["pose_enc"] = [pose_enc_list[-1]]  # pose encoding of the last iteration
                    predictions["pose_enc_list"] = [pose_enc_list]
                    
                    predictions["per_chunk_pose_enc"] = chunk_pose_enc #tensor of B,1,7
                    predictions["per_frame_pose_enc"] = frame_pose_enc #tensor of B,S-1,7

                    if self.enable_memory:
                        predictions["global_tokens"] = [global_tokens]
                else:
                    
                    context.setdefault("pose_enc", []).append(pose_enc_list[-1])
                    predictions["pose_enc"] = context["pose_enc"]
                    context.setdefault("pose_enc_list", []).append(pose_enc_list)
                    predictions["pose_enc_list"] = context["pose_enc_list"]

                    predictions["per_chunk_pose_enc"] = merge_results(context["per_chunk_pose_enc"], chunk_pose_enc, 0 , 1)
                    predictions["per_frame_pose_enc"] = merge_results(context["per_frame_pose_enc"], frame_pose_enc, 0 , 1)
                    
                    if self.enable_memory:
                        context.setdefault("global_tokens", []).append(global_tokens)
                        predictions["global_tokens"] = context["global_tokens"]
                    #else:
                    #    predictions["global_tokens"] = [global_tokens]

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    filtered_aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )

                #apply scale correction
                depth *= chunk_scale.view(B,1,1,1,1)

                if context is None:
                    predictions["depth"] = [depth]
                    predictions["depth_conf"] = [depth_conf]
                    
                else:
                    context.setdefault("depth", []).append(depth)
                    predictions["depth"] = context["depth"]
                    context.setdefault("depth_conf", []).append(depth_conf)
                    predictions["depth_conf"] = context["depth_conf"]
                        
                        

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    filtered_aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )

                #align point maps using last stage transform pose enc
                if self.camera_head is not None:

                    if context is not None:
                        #use first pose of chunk, since points are given in its coordinate frame
                        point_transform = closed_form_inverse_se3(per_frame_extr[:, 0]).unsqueeze(1)  # (B, 1, 4, 4)
                    
                        #apply identity alignment
                        point_transform = point_transform @ point_identity_alignment.view(B,1,4,4)
                    else:
                        point_transform = point_identity_alignment.view(B,1,4,4)

                    #apply scale correction
                    pts3d*= chunk_scale.view(B,1,1,1,1)

                    #convert points to homogeneous coordinates
                    pts3d_h = torch.cat([pts3d, torch.ones_like(pts3d[..., :1])], dim=-1) # (B, S, H, W, 4)

                    pts3d_h = pts3d_h.view(B, -1, 4)  # (B, S*H*W, 4)
                    point_transform = point_transform.expand(-1, pts3d_h.size(1), -1, -1)  # (B, S*H*W, 4, 4)

                    pts3d_transformed = torch.matmul(point_transform, pts3d_h.unsqueeze(-1)).squeeze(-1)  # (B, S*H*W, 4)

                    pts3d = pts3d_transformed[..., :3].view(B,S,H,W,3) # (B, S, H, W, 3)

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
                    predictions["images"] = [images]  # store the images for visualization during inference
                else:
                    context.setdefault("images", []).append(images)
                    predictions["images"] = context["images"]
        
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

    #normalize just to be sure
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

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

    if isinstance(first_chunk, list) and isinstance(second_chunk, list):

        if num_overlap > 0:
            second_chunk = [item[:,num_overlap:] for item in second_chunk]
    
        merged = [torch.cat((fc_item, sc_item), dim=mergeDim) for fc_item, sc_item in zip(first_chunk, second_chunk)]
    else:
        if num_overlap > 0:
            second_chunk = second_chunk[:,num_overlap:]
        
        merged = torch.cat((first_chunk, second_chunk), dim=mergeDim)

    del first_chunk
    del second_chunk

    return merged
