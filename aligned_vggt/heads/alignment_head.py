import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Tuple, List

from vggt.vggt.layers.block import Block
from vggt.vggt.layers import Mlp

from aligned_vggt.layers.cross_attention import CrossAttentionBlock
from aligned_vggt.layers.rope import RotaryPositionEmbedding
from aligned_vggt.layers.gated_update import GatedUpdate
from vggt.vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter

logger = logging.getLogger(__name__)


class AlignmentHead(nn.Module):
    """
    Head that yields per-frame similarity alignments given features from the VGGT backbone of the current and previous chunks.
    The head processes the features through alternating frame and temporal attention blocks, followed by a decoder to generate alignment predictions.
    After appending an token for capturing per-frame alignment information, the features of the current chunk are processed through alternating frame and temporal attention blocks.
    Within the frame attention blocks, each frame's features attend to themselves spatially, while in the temporal attention blocks, each frame's features attend to the features of the same spatial location in overlapping frames from the previous chunk.
    The contextualized per-frame tokens of the current chunk are then decoded to yield per-frame similarity transformations.
    The first token of the current chunk is used to predict a chunk-level similarity transformation, whereas the remaining tokens predict per-frame rigid transformations relative to the chunk-level transformation.
    If enabled, memory tokens are used to capture global alignment information across chunks and inject past information for better temporal consistency.


    Args:
        patch_size (int): Size of the image patches.
        in_dim (int): Input feature dimension.
        embed_dim (int): Embedding dimension for attention blocks.
        dec_dim (int): Dimension for the decoder.
        depth_aa (int): Number of alternating attention blocks.
        depth_decoder (int): Number of decoder blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio for MLP hidden dimension.
        num_register_tokens (int): Number of register tokens.
        qkv_bias (bool): Whether to use bias in QKV projections.
        proj_bias (bool): Whether to use bias in output projection.
        ffn_bias (bool): Whether to use bias in feed-forward network.
        aa_order (list[str]): Order of attention types in alternating attention blocks, e.g. ["frame", "temporal"].
        aa_block_size (int): Number of consecutive blocks of the same attention type.
        qk_norm (bool): Whether to normalize Q and K vectors.
        rope_freq (int): Frequency for rotary position embeddings. -1 to disable.
        init_values (float): Initial value for LayerScale.
        num_memory_tokens (int): Number of memory tokens.
        temporal_attention (bool): Whether to use temporal attention blocks.
    """

    def __init__(
        self,
        patch_size=14,
        in_dim=2048, # higher than in aggregator since frame and global attention outputs are concatenated
        embed_dim = 1024,
        dec_dim = 512,
        depth_aa=4,
        depth_decoder=2,
        num_heads= 8,
        mlp_ratio=4.0,
        num_register_tokens=4,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        aa_order=["frame", "temporal"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        num_memory_tokens=8,
        temporal_attention = True,
    ):
        super().__init__()

        self.num_memory_tokens = num_memory_tokens
        self.temporal_attention = temporal_attention

        self.depth_aa = depth_aa
        self.aa_order = aa_order
        self.aa_block_size = aa_block_size
        self.patch_size = patch_size

        # Validate that depth is divisible by aa_block_size
        if self.depth_aa % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth_aa}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth_aa // self.aa_block_size

        self.depth_decoder = depth_decoder

        # The patch tokens start after the pose, camera, and register tokens
        self.patch_start_idx = 1 + 1 + num_register_tokens

        # dropout parameters
        self.drop_prob_nonoverlap = 0.2

        self.use_reentrant = False # hardcoded to False

        self.project_in = nn.Linear(in_dim, embed_dim)
        self.project_dec = nn.Linear(embed_dim, dec_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope1d = RotaryPositionEmbedding(frequency=rope_freq) if rope_freq > 0 else None

        # Initialize rotary position embedding if frequency > 0
        self.rope2d = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope2d is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope2d,
                )
                for _ in range(depth_aa)
            ]
        )

        if temporal_attention:
            self.temporal_blocks = nn.ModuleList(
                [
                    CrossAttentionBlock(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        init_values=init_values,
                        qk_norm=qk_norm,
                        rope=self.rope1d,
                    )
                    for _ in range(depth_aa)
                ]
            )
        else:
        
            aa_order=["frame", "global"]
            self.global_blocks = nn.ModuleList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        init_values=init_values,
                        qk_norm=qk_norm,
                        rope=self.rope2d,
                    )
                    for _ in range(depth_aa)
                ]
            )

        
        self.chunk_cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=dec_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope1d,
                )
                for _ in range(depth_decoder)
            ]
        )


        self.frame_cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=dec_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope1d,
                )
                for _ in range(depth_decoder)
            ]
        )

        self.chunk_sim3_decoder = Mlp(in_features=dec_dim, hidden_features=dec_dim // 2, out_features=8, drop=0)
        self.frame_se3_decoder = Mlp(in_features=dec_dim, hidden_features=dec_dim // 2, out_features=7, drop=0)

        self.token_norm = nn.LayerNorm(embed_dim)
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.chunk_norm = nn.LayerNorm(dec_dim)
        self.frame_norm = nn.LayerNorm(dec_dim)

        self.per_frame_alignment_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        nn.init.normal_(self.per_frame_alignment_token, std=1e-6) # Initialize parameters with small values

        if self.num_memory_tokens > 0:
            self.memory_token = nn.Parameter(torch.empty(1, num_memory_tokens, dec_dim))
            nn.init.orthogonal_(self.memory_token[0]) # avoid early collapse
            self.memory_token.data = F.normalize(self.memory_token.data, dim=-1) # ensures unit norm

            # additional stuff for hybrid initialization
            self.frame_proj = nn.Linear(dec_dim, num_memory_tokens * dec_dim)
            self.alpha = nn.Parameter(torch.tensor(0.1)) # learnable factor for hybrid initialization

            # gated update module for global pose token
            self.gated_update = GatedUpdate(dec_dim, num_memory_tokens)


    def forward(self, tokens: torch.Tensor, image_size: Tuple[int, int], next_num_overlap: int, overlap_tokens : torch.Tensor = None, memory_tokens : torch.Tensor = None) -> Tuple[List[torch.Tensor], int]:
        """
            Forward pass for the alignment head.
            Args:
                tokens (torch.Tensor): Input tokens of shape (B, S, P, C).
                image_size (Tuple[int, int]): Size of the input images (H, W).
                next_num_overlap (int): Number of overlapping frames for the next chunk.
                overlap_tokens (torch.Tensor, optional): Overlapping tokens from the previous chunk of shape (B, T, P, C).
                memory_tokens (torch.Tensor, optional): Memory tokens from the previous chunk of shape (B, num_memory_tokens, C).
            Returns:
                chunk_sim3_enc (torch.Tensor): Predicted chunk similarity transformation of shape (B, 1, 8).
                frame_se3_encs (torch.Tensor): Predicted per-frame rigid transformations of shape (B, S-1, 7).
                memory_tokens (torch.Tensor): Updated memory tokens of shape (B, num_memory_tokens, C).
                new_overlap_tokens (torch.Tensor): Processed overlapping tokens for the next chunk of shape (B, T, P, C).
        """
        H, W = image_size

        # project to lower dim
        tokens = self.project_in(tokens)

        B, S, P, C = tokens.shape  # B: batch size, S: sequence length, P: number of tokens, C: channels

        # normalize tokens
        tokens = self.token_norm(tokens)
        
        if overlap_tokens is not None:
            # 1+p, since cross tokens already have pose token
            assert overlap_tokens.shape[0] == B and overlap_tokens.shape[2]==1 + P and overlap_tokens.shape[3]==C, "Size of tokens and overlap tokens must match"

            T = overlap_tokens.shape[1] #previous overlap + first token

            # move cross attention tokens to device
            if overlap_tokens.device != tokens.device:
                overlap_tokens.to(tokens.device)

            # detach to improve stability
            overlap_tokens = overlap_tokens.detach()

            # no normalization necessary, as they are normalized in cross attention blocks
            first_chunk = False
        else:
            T = None
            first_chunk = True

        # prepare alignment tokens
        alignment_tokens = slice_expand_and_flatten(self.per_frame_alignment_token,B,S)
        tokens = torch.cat([alignment_tokens, tokens], dim=2)

        # update P because we added special tokens
        _, _, P, C = tokens.shape

        # 1d pos enc for tokens and cross tokens (same enc for overlapping frames)
        pos_temporal = None
        if self.rope1d is not None:
            if self.temporal_attention:
                seq_ids = torch.arange(S,device=tokens.device)
                if overlap_tokens is not None:
                    att_ids = seq_ids + (S-(T-1))
                    cross_ids = torch.cat([seq_ids[:1],seq_ids[-(T-1):]])
                    pos_temporal = att_ids.view(1, S).expand(B * P, -1), cross_ids.view(1, T).expand(B * P, -1)
                else:
                    pos_temporal = seq_ids.view(1, S).expand(B * P, -1), seq_ids.view(1, S).expand(B * P, -1)
            else:
                if self.rope2d is not None:
                    if overlap_tokens is not None:
                        pos_temporal = self.position_getter(B * (S+T), H // self.patch_size, W // self.patch_size, device=tokens.device)
                    else:
                        pos_temporal = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=tokens.device)

                    if self.patch_start_idx > 0:
                        # do not use position embedding for special tokens (camera and register tokens)
                        # so set pos to 0 for the special tokens
                        pos_temporal = pos_temporal + 1
                        pos_temporal_special = torch.zeros(pos_temporal.shape[0], self.patch_start_idx, 2).to(tokens.device).to(pos_temporal.dtype)
                        pos_temporal = torch.cat([pos_temporal_special, pos_temporal], dim=1)

        # 2d pose enc (frame attention)
        pos2d = None
        if self.rope2d is not None:
            pos2d = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=tokens.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos2d = pos2d + 1
            pos2d_special = torch.zeros(B * S, self.patch_start_idx, 2).to(tokens.device).to(pos2d.dtype)
            pos2d = torch.cat([pos2d_special, pos2d], dim=1)


        frame_idx = 0
        temporal_idx = 0
        global_idx = 0

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:

                # do one 1d attention over time (cross features: previous chunk overlapping + current chunk) and on per frame attention
                if attn_type == "frame":
                    tokens, frame_idx = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos2d
                    )
                elif attn_type == "temporal":
                    
                    tokens, temporal_idx = self._process_temporal_attention(
                        tokens, B, S, P, C, temporal_idx, cross_tokens=overlap_tokens, T=T, pos=pos_temporal
                    )
                elif attn_type == "global":
                    tokens, global_idx = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos_temporal, overlap_tokens=overlap_tokens, T=T
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

        tokens = tokens.view(B,S,P,C)
        per_frame_alignment_tokens = tokens[..., 0, :]
        
        with torch.amp.autocast("cuda", enabled=False):
            chunk_sim3_enc, frame_se3_encs, memory_tokens = self._decode_alignments(per_frame_alignment_tokens, next_num_overlap, first_chunk, memory_tokens=memory_tokens)
        
        new_overlap_tokens = torch.cat([tokens[:,:1] , tokens[:, -next_num_overlap:]],dim=1)

        return chunk_sim3_enc, frame_se3_encs, memory_tokens, new_overlap_tokens.contiguous() # only return processed overlap tokens
        
    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks.
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1

        return tokens, frame_idx

    def _process_temporal_attention(self, tokens, B, S, P, C, temporal_idx, pos=None, cross_tokens = None, T = None):
        """
        Process temporal attention blocks.
        """
        if tokens.shape != (B * P, S, C):
            tokens = tokens.view(B, S, P, C).view(B * P, S, C)

        if cross_tokens is not None:
            if cross_tokens.shape != (B * P, T, C):
                cross_tokens = cross_tokens.view(B, T, P, C).view(B * P, T, C)
        else:
            # only do time-aware self attention
            cross_tokens = tokens
        
        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.temporal_blocks[temporal_idx], tokens, cross_tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.temporal_blocks[temporal_idx](tokens,cross_tokens, pos=pos)
            temporal_idx += 1

        return tokens, temporal_idx
    
    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, overlap_tokens = None, T = None):
        """
        Process global attention blocks.
        """

        # if we have overlap tokens, we need to concat them first
        if overlap_tokens is not None:
            if overlap_tokens.shape != (B, T, P, C):
                overlap_tokens = overlap_tokens.view(B, T, P, C)

            if tokens.shape != (B, S, P, C):
                tokens = tokens.view(B, S, P, C)
            tokens = torch.cat((overlap_tokens, tokens), dim=1)  # (B, T+S, P, C)
            S = S + T

        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1

        if T is not None:
            # remove overlap tokens again
            tokens = tokens.view(B, S, P, C)[:, T:, : , :].contiguous().view(B, (S - T) * P, C)

        return tokens, global_idx

    def _decode_alignments(self, frame_alignment_tokens : torch.Tensor, num_overlap: int, is_first_chunk: bool, memory_tokens : torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode alignments from frame alignment tokens.
        Args:
            frame_alignment_tokens (torch.Tensor): Frame alignment tokens of shape (B, S, C).
            num_overlap (int): Number of overlapping frames.
            is_first_chunk (bool): Whether this is the first chunk.
            memory_tokens (torch.Tensor, optional): Memory tokens from the previous chunk of shape (B, num_memory_tokens, C).
        Returns:
            chunk_sim3_enc (torch.Tensor): Predicted chunk similarity transformation of shape (B, 1, 8).
            frame_se3_encs (torch.Tensor): Predicted per-frame rigid transformations of shape (B, S-1, 7).
            memory_tokens (torch.Tensor): Updated memory tokens of shape (B, num_memory_tokens, C).
        """
        B,S,C = frame_alignment_tokens.shape

        # 1d pos enc
        pos1d_frame_cross = None
        pos1d_cross = None
        if self.rope1d is not None:
            seq_ids = torch.arange(1,S,device=frame_alignment_tokens.device) #all besides first frame
            cross_ids = torch.zeros(1,device=frame_alignment_tokens.device,dtype=seq_ids.dtype) #first frame
            pos1d_frame_cross = seq_ids.view(1, (S-1)).expand(B, -1), cross_ids.view(1, 1).expand(B, -1)
            
            if self.num_memory_tokens > 0:
                cross_ids = torch.arange(0,S+self.num_memory_tokens,device=frame_alignment_tokens.device) #0,...,S+num_memory_tokens
                cross_ids[-self.num_memory_tokens:] += S #shift memory token, so we have a unique position outside the normal scope of frames => 0,...,S-1,2S

                att_ids = torch.zeros(1,device=frame_alignment_tokens.device,dtype=cross_ids.dtype)
                pos1d_cross = att_ids.view(1, 1).expand(B, -1), cross_ids.view(1, (S+self.num_memory_tokens)).expand(B, -1)
            else:
                cross_ids = torch.arange(0,S,device=frame_alignment_tokens.device)
                att_ids = torch.zeros(1,device=frame_alignment_tokens.device,dtype=cross_ids.dtype)
                pos1d_cross = att_ids.view(1, 1).expand(B, -1), cross_ids.view(1, S).expand(B, -1)


        # prepare alignment tokens
        tokens = self.project_dec(frame_alignment_tokens)
        B,S,C = tokens.shape # update shape
        tokens = self.dec_norm(tokens)

        # prepare memory tokens
        if self.num_memory_tokens > 0:
            normalized_tokens = tokens.norm(dim=-1).mean(dim=-1, keepdim=True).unsqueeze(1)  # (B,1,1)
            # init memory tokens
            if memory_tokens is None:
                memory_tokens = self.memory_token.expand(B,*self.memory_token.shape[1:])

                # hybrid init from projection of first chunk token
                frame_init = self.frame_proj(tokens[:,0]).view(B,-1,C)
                frame_dir = frame_init / frame_init.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                alpha_clamped = torch.sigmoid(self.alpha)
                directional_memory = (1-alpha_clamped) * memory_tokens + alpha_clamped * frame_dir
                effective_memory = memory_tokens * normalized_tokens
            else:

                # can't detach here, because if done, no gradient will be accumulated for memory updates
                # memory_tokens = memory_tokens #.detach()
                directional_memory = memory_tokens
                effective_memory = memory_tokens * normalized_tokens
            
            assert directional_memory.shape[0] == B, "Memory tokens must have same batch dimension as frame tokens"

            cross_tokens = torch.cat([tokens,effective_memory],dim=1)
        else:
            cross_tokens = tokens

        # aggregate alignment information for whole chunk
        first_frame_alignment_token = tokens[:,:1]
        chunk_cross_idx = 0
        for _ in range(self.depth_decoder):
            if self.training:
                first_frame_alignment_token = checkpoint(self.chunk_cross_blocks[chunk_cross_idx], first_frame_alignment_token, cross_tokens, pos1d_cross, use_reentrant=self.use_reentrant)
            else:
                first_frame_alignment_token = self.chunk_cross_blocks[chunk_cross_idx](first_frame_alignment_token,cross_tokens, pos=pos1d_cross)
            chunk_cross_idx += 1

        # memory update
        if self.num_memory_tokens > 0:
            # compute updated memory token
            memory_tokens = self.gated_update(directional_memory,first_frame_alignment_token) # (B, num_memory_tokens, embed_dim)
        updated_chunk_alignment_token = self.chunk_norm(first_frame_alignment_token) # (B, 1, embed_dim)
        
        # process per frame tokens (excluding first frame)
        frame_tokens = tokens[:,1:]
        # frame dropout (avoid dropping overlapping frames, avoid dropout for first chunks in sequence and first frame in general)
        if self.training and self.drop_prob_nonoverlap > 0.0 and not is_first_chunk and (S-1-num_overlap) > 1:
            non_overlap_mask = (torch.rand(B, S-1-num_overlap, device=tokens.device) > self.drop_prob_nonoverlap).float().unsqueeze(-1) # (B, S-num_overlap, 1)
            overlap_mask = torch.ones((B, num_overlap, 1), device=tokens.device)
            mask = torch.cat((non_overlap_mask,overlap_mask),dim=1)

            scale = 1.0 / (1.0 - self.drop_prob_nonoverlap) #rescale so expected value remains the same
            
            frame_tokens = frame_tokens * mask * scale
        else:
            frame_tokens = frame_tokens

        # inform frame tokens over chunk pose alignment
        frame_cross_idx = 0
        for _ in range(self.depth_decoder):
            if self.training:
                frame_tokens = checkpoint(self.frame_cross_blocks[frame_cross_idx], frame_tokens, updated_chunk_alignment_token, pos1d_frame_cross, use_reentrant=self.use_reentrant)
            else:
                frame_tokens = self.frame_cross_blocks[frame_cross_idx](frame_tokens,updated_chunk_alignment_token, pos=pos1d_frame_cross)
            frame_cross_idx += 1
        frame_tokens = self.frame_norm(frame_tokens)

        # decode per frame pose 
        frame_se3_encs = self.frame_se3_decoder(frame_tokens) # (B, S-1, 7)

        # decode chunk sim3 
        chunk_sim3_enc = self.chunk_sim3_decoder(updated_chunk_alignment_token) # (B, 1, 8)
        chunk_sim3_enc[:,:,-1] = torch.exp(chunk_sim3_enc[:,:,-1]) # map scale

        return chunk_sim3_enc, frame_se3_encs, memory_tokens
        

def slice_expand_and_flatten(token_tensor : torch.Tensor, B: int, S: int) -> torch.Tensor:
    """
    Adapted from VGGT codebase.
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens

    Args:
        token_tensor (torch.Tensor): Specialized tokens with shape (1, 2, X, C).
        B (int): Batch size.
        S (int): Sequence length.
    Returns:
        torch.Tensor: Processed tokens with shape (B, S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)
    
    return combined