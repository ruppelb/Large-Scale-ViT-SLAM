import torch
import random
from vggt.utils.rotation import quat_to_mat, mat_to_quat
from aligned_vggt.utils.alignment import * #per_chunk_scale_alignment_from_poses, scale_alignment_from_poses, apply_sim3_alignment_on_dict, umeyama_alignment_from_poses, umeyama_alignment_from_points


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

def convertDictListsToTensors(chunked_dict, overlap, out_dict=None):

        #convert dict lists (batch and gt) to tensors (if they are lists to begin with)
        #if validating, remove double overlapping frames
        if out_dict == None:
            out_dict = chunked_dict

        keys_to_merge = ["pose_enc", "pose_enc_list", "world_points", "world_points_conf", "depth", "depth_conf", "extrinsics", "intrinsics", "scales", "cam_points", "depths", "point_masks", "images", "ids"]

        for key in chunked_dict.keys():
            if key in keys_to_merge:
                if isinstance(chunked_dict[key][0],list):
                    #pose encoding list

                    if overlap > 0:
                        #omit overlapping frames
                        for i in range(1,len(chunked_dict[key])):
                            chunked_dict[key][i] = [item[:,overlap:] for item in chunked_dict[key][i]]

                    out_dict[key] = [torch.cat(tensor_tuple, dim=1) for tensor_tuple in zip(*chunked_dict[key])]
                else:

                    if overlap > 0:
                        #omit overlapping frames
                        for i in range(1,len(chunked_dict[key])):
                            chunked_dict[key][i] = chunked_dict[key][i][:,overlap:]

                    out_dict[key] = torch.cat(chunked_dict[key],dim=1)

def moveDictListItemToCPU(chunked_dict,itemIndex):

    for key in chunked_dict.keys():
        if isinstance(chunked_dict[key],list):
            if len(chunked_dict[key]) >= (abs(itemIndex) if itemIndex < 0 else itemIndex+1):
                if isinstance(chunked_dict[key][0],list):
                    #pose encoding list
                    chunked_dict[key][itemIndex] = [(item.cpu() if isinstance(item, torch.Tensor) else item) for item in chunked_dict[key][itemIndex]]
                else:
                    if isinstance(chunked_dict[key][itemIndex], torch.Tensor):
                        chunked_dict[key][itemIndex] = chunked_dict[key][itemIndex].cpu()



def alignAndConvertOutputs(predictions, batch, chunked_batch, alignment_type, seq_width, overlap):

        #perform gt and pred alignment
        if alignment_type == "per_chunk_scale_from_poses":
            #for this we need chunked outputs
            per_chunk_scale_alignment_from_poses(predictions,chunked_batch)
            convertDictListsToTensors(chunked_batch,overlap,batch)
            convertDictListsToTensors(predictions,overlap)
            #scale_alignment_from_poses(predictions,batch) #final full scale alignment
        else:
            convertDictListsToTensors(chunked_batch,overlap,batch)
            convertDictListsToTensors(predictions,overlap)
            if alignment_type == "scale_from_fc_poses":
                scale_alignment_from_poses(predictions,batch,seq_width)
            elif alignment_type == "scale_from_poses":
                scale_alignment_from_poses(predictions,batch)
            elif alignment_type == "per_frame_scale_from_poses":
                per_frame_scale_alignment_from_poses(predictions,batch)
            elif alignment_type == "sim3_from_poses":
                umeyama_alignment_from_poses(predictions, batch, seq_width)
            elif alignment_type == "sim3_from_points":
                
                if "world_points" not in predictions:
                    raise ValueError("sim3_from_points alignment requires point head to be enabled.")
                
                batch_transforms, batch_scales = umeyama_alignment_from_points(predictions["world_points"][:,:seq_width], predictions["world_points_conf"][:,:seq_width], batch["world_points"][:,:seq_width], batch["point_masks"][:,:seq_width], confidence_threshold=50.0) #90
                apply_sim3_alignment_on_dict(predictions, batch["images"].shape[-2:], batch_transforms, batch_scales)
            elif alignment_type == "scale_from_depths":
                scale_align_from_depths(predictions,batch)
            else:
                #no alignment
                #pass
                
                #if network outputs global scale, apply here
                if "global_scales" in predictions:
                    B = batch["extrinsics"].shape[0]
                    for b in range(B):
                        batch_scale = predictions["global_scales"][b].to(predictions["pose_enc_list"][-1])

                        #inplace operations
                        predictions["pose_enc_list"][-1][b,:,:3] *= batch_scale

                        if "depth" in predictions:
                            predictions["depth"][b,...] *= batch_scale

                        if "world_points" in predictions:
                            predictions["world_points"][b,...] *= batch_scale

                    predictions["pose_enc"] = predictions["pose_enc_list"][-1]
                


def generate_chunks(num_frames, mode, seq_width, overlap):
    indices = []
    if mode == "chunk_gt":

        #chunk sequence in non-overlapping sequences of width seq_width
        for i in range(0, num_frames - seq_width + 1, seq_width):
            indices.append(list(range(i,i+seq_width)))

        #check if all images are at least within one sequence  
        if len(indices) * seq_width < num_frames:
            indices.append(list(range(len(indices) * seq_width, num_frames)))

    elif mode == "chunk_overlap":

        if num_frames < seq_width:
            indices.append(list(range(num_frames)))
        else:
            #prepare overlapping sequences
            for i in range(0, num_frames - seq_width + 1, seq_width - overlap):
                indices.append(list(range(i,i+seq_width)))
                
            #check if all images are at least within one sequence  
            if len(indices) * (seq_width - overlap) < num_frames - overlap:
                #create a subsequence with the last images
                indices.append(list(range(len(indices) * (seq_width - overlap), num_frames)))
    elif mode == "all":
        indices = [list(range(num_frames))]
    elif mode == "two_chunks":
        #sample two non-overlapping chunks regardless of seq_width
        if num_frames < 2:
            raise ValueError("Number of frames must be at least 2 for two_chunks mode.")
        elif num_frames == 2:
            indices = [[0,1]]
        else:
            #mid_point = num_frames // 2
            #indices = [list(range(0, mid_point)), list(range(mid_point, num_frames))]
            #random sample two non-overlapping sets from num_frames
            all_indices = list(range(num_frames))
            first_chunk_size = random.randint(1, num_frames - 1)
            first_chunk = random.sample(all_indices, first_chunk_size)
            second_chunk = [idx for idx in all_indices if idx not in first_chunk]
            indices = [first_chunk, second_chunk]
    else:
        raise ValueError(f"Unknown sequence generation mode: {mode}")
    return indices

def chunk_batch(batch, indices):
    #generate chunked batch data
    chunked_batch = {}
    for chunk_ids in indices:
        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor):
                chunked_batch.setdefault(key,[]).append(batch[key][:,chunk_ids])
    return chunked_batch

"""
#generate chunk indices      
indices = [] 
for i in range(0, S - self.seq_width + 1, self.seq_width - self.num_overlap):
    indices.append(list(range(i,i+self.seq_width)))
    
#check if all images are at least within one sequence  
if len(indices) * (self.seq_width - self.num_overlap) < S - self.num_overlap:
    #create a subsequence with the last images
    indices.append(list(range(len(indices) * (self.seq_width - self.num_overlap), S)))
"""