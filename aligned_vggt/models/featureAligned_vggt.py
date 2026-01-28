import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.vggt.models.aggregator import Aggregator
from vggt.vggt.heads.camera_head import CameraHead
from vggt.vggt.heads.dpt_head import DPTHead
from vggt.vggt.heads.track_head import TrackHead
from vggt.vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from vggt.vggt.utils.geometry import closed_form_inverse_se3

from aligned_vggt.heads.alignment_head import AlignmentHead
from aligned_vggt.utils.data import extri_to_pose_encoding, pose_encoding_to_extri
from aligned_vggt.utils.geometry import averagePoseEncodings

class FeatureAlignedVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True, num_memory_tokens = 8, temporal_attention = True):
        super().__init__()

        self.embed_dim = embed_dim
        self.enable_memory = num_memory_tokens > 0

        self.intermediate_layer_indices = [4, 11, 17, 23] # indices of intermediate layers
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1", intermediate_layer_idx=range(len(self.intermediate_layer_indices))) if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1", intermediate_layer_idx=range(len(self.intermediate_layer_indices))) if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

        self.alignment_head = AlignmentHead(in_dim=2*embed_dim,patch_size=patch_size,num_memory_tokens= num_memory_tokens,temporal_attention=temporal_attention)

    def set_config(self, cfg: dict):
        """
        Set the configuration for the FeatureAlignedVGGT model. (Called after loading from PyTorchModelHubMixin)
        """
        
        self.camera_head = self.camera_head if cfg.enable_camera else None
        self.point_head = self.point_head if cfg.enable_point else None
        self.depth_head = self.depth_head if cfg.enable_depth else None
        self.track_head = self.track_head if cfg.enable_track else None

        self.enable_memory = cfg.num_memory_tokens > 0

        self.alignment_head = AlignmentHead(in_dim=2*self.embed_dim,patch_size=cfg.patch_size,num_memory_tokens=cfg.num_memory_tokens,temporal_attention=cfg.temporal_attention)

    def forward(self, images: torch.Tensor, num_overlap: int, context: dict = None, gt_poses: torch.Tensor = None) -> dict:
        """
        Forward pass of the FeatureAlignedVGGT model.
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
                - 'chunk_sim3_alignment_enc': Tensor of shape (B, 1, 7) containing the sim3 chunk alignment encoding.
                - 'frame_se3_alignment_enc': Tensor of shape (B, S-1, 7) containing the se3 per-frame alignment encodings.
                - 'depth': List of depth maps for all processed chunks.
                - 'depth_conf': List of depth confidence maps for all processed chunks.
                - 'world_points': List of 3D point maps for all processed chunks.
                - 'world_points_conf': List of 3D point confidence maps for all processed chunks
                - 'overlap_tokens': Overlap tokens for context propagation.
                - 'memory_tokens': List of memory tokens for all processed chunks (if memory is enabled).
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

        # get context tokens if available
        context_overlap_tokens = None
        context_global_tokens = None
        if context is not None:
            context_overlap_tokens = context["overlap_tokens"]
            if self.enable_memory:
                context_global_tokens = context["memory_tokens"][-1]

        # compute chunk and frame alignments
        overlap = num_overlap if S > num_overlap else S-1 # handle case that we get a chunk smaller than num overlap (only possible when we sample a single chunk during inference)
        chunk_sim3_enc, frame_se3_enc, memory_tokens, overlap_tokens = self.alignment_head(filtered_aggregated_tokens_list[-1], (H, W), overlap, overlap_tokens=context_overlap_tokens, memory_tokens=context_global_tokens)

        # compute final per-frame SE3 transforms and chunk scale
        chunk_se3 = pose_encoding_to_extri(chunk_sim3_enc) # (B, 1, 4, 4)
        chunk_scale = chunk_sim3_enc[...,-1] # (B, 1)
        per_frame_se3 = pose_encoding_to_extri(frame_se3_enc) # (B, S-1, 4, 4)
        per_frame_se3 = torch.matmul(per_frame_se3, chunk_se3)
        per_frame_se3 = torch.cat([chunk_se3,per_frame_se3],dim=1) # (B, S, 4, 4)

        # decode poses, depth, and points maps for each chunk separately
        with torch.amp.autocast("cuda", enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(filtered_aggregated_tokens_list)

                # decode final vggt pose outputs
                extr, intr = pose_encoding_to_extri_intri(pose_enc_list[-1],image_size_hw=images.shape[-2:]) # (B, S, 3, 4)
                extr = torch.nn.functional.pad(extr, (0,0,0,1,0,0,0,0), mode="constant")
                extr[:,:, 3, 3] = 1.

                # assure first pose is identity
                extr_identity_alignment = closed_form_inverse_se3(extr[:,0]) # (B, 4, 4)
                point_identity_alignment = extr[:,0].detach().clone()
                extr = extr @ extr_identity_alignment.view(B,1,4,4)

                # apply scale correction
                extr[:,:,:3,3] *= chunk_scale.view(B,1,1)

                # get initial chunk alignment estimate if context is given
                if context is not None:
                    if gt_poses is not None:
                        mean_camera_transform = gt_poses[:,:1].to(extr) # (B, 1, 4, 4)
                    else:
                        context_overlapping = pose_encoding_to_extri(context["pose_enc"][-1][:, -overlap:]) # (B, N_overlap, 4, 4)
                        inv_overlapping = closed_form_inverse_se3(extr[:, :overlap].reshape(B*overlap,4,4)).reshape(B,overlap,4,4)
                        camera_transforms = inv_overlapping @ context_overlapping

                        if overlap > 1:
                            camera_transforms = extri_to_pose_encoding(camera_transforms)
                            mean_camera_transform = averagePoseEncodings(camera_transforms) # (B, 1, 7)
                            mean_camera_transform = pose_encoding_to_extri(mean_camera_transform) # (B, 1, 4, 4)
                        else:
                            mean_camera_transform = camera_transforms
                else:
                    mean_camera_transform = torch.eye(4, device=images.device, dtype=images.dtype).view(1,1,4,4).expand(B,-1,-1,-1)
                    
                per_frame_se3 = torch.matmul(per_frame_se3, mean_camera_transform)

                # apply final pose alignment
                aligned_extr = torch.matmul(extr, per_frame_se3)
                aligned_pose_enc = extri_intri_to_pose_encoding(aligned_extr, intr, image_size_hw=images.shape[-2:])

                # save dict keys
                predictions["overlap_tokens"] = overlap_tokens

                if context is None:
                    predictions["pose_enc"] = [aligned_pose_enc]  # pose encoding of the last iteration
                    predictions["chunk_sim3_alignment_enc"] = chunk_sim3_enc #tensor of (B, 1, 7)
                    predictions["frame_se3_alignment_enc"] = frame_se3_enc #tensor of (B, S-1, 7)

                    if self.enable_memory:
                        predictions["memory_tokens"] = [memory_tokens]
                else:
                    context.setdefault("pose_enc", []).append(aligned_pose_enc)
                    predictions["pose_enc"] = context["pose_enc"]
                    predictions["chunk_sim3_alignment_enc"] = merge_results(context["chunk_sim3_alignment_enc"], chunk_sim3_enc, 0 , 1)
                    predictions["frame_se3_alignment_enc"] = merge_results(context["frame_se3_alignment_enc"], frame_se3_enc, 0 , 1)

                    if self.enable_memory:
                        context.setdefault("memory_tokens", []).append(memory_tokens)
                        predictions["memory_tokens"] = context["memory_tokens"]

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    filtered_aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )

                # apply scale correction
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

                # align point maps using per-frame SE3 transforms
                if self.camera_head is not None:

                    if context is not None:
                        # use first pose of chunk, since points are given in its coordinate frame
                        point_transform = closed_form_inverse_se3(per_frame_se3[:, 0]).unsqueeze(1)  # (B, 1, 4, 4)
                        point_transform = point_transform @ point_identity_alignment.view(B,1,4,4) # apply identity alignment
                    else:
                        point_transform = point_identity_alignment.view(B,1,4,4)

                    # apply scale correction
                    pts3d*= chunk_scale.view(B,1,1,1,1)

                    # convert points to homogeneous coordinates
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

def merge_results(first_chunk, second_chunk, num_overlap : int = 0, mergeDim : int = 1):
    """
    Merge two chunks of results, removing overlapping parts from the second chunk.
    Args:
        first_chunk: The first chunk of results (can be a tensor or a list of tensors).
        second_chunk: The second chunk of results (can be a tensor or a list of tensors).
        num_overlap (int): Number of overlapping elements to remove from the start of the second chunk.
        mergeDim (int): Dimension along which to concatenate the chunks.
    Returns:
        Merged result (same type as input chunks).
    """

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
