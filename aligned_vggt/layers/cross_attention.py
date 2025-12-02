import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from typing import Callable

from vggt.vggt.layers.layer_scale import LayerScale
from vggt.vggt.layers.mlp import Mlp
from vggt.vggt.layers.attention import Attention

XFORMERS_AVAILABLE = False


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        #self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, y: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        
        assert x.shape[0] == y.shape[0] and x.shape[2] == y.shape[2], "x and y must have the same shape in batch and token dim"
        M = y.shape[1]

        pos_q, pos_k = pos

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(y).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(y).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q = self.rope(q, pos_q)
            k = self.rope(k, pos_k)
        
        #adjusted due to crashes in torch 2.5.1, see https://github.com/pytorch/pytorch/issues/150054
        if self.fused_attn and B < 7000:
            mask = torch.ones((N, M), dtype=torch.bool, device=q.device)
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, attn_mask=mask)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm3 = norm_layer(dim)

    def forward(self, x: Tensor, y: Tensor, pos=None) -> Tensor:
        
        x = x + self.ls1(self.attn(self.norm1(x), self.norm3(y), pos=pos))
        x = x + self.ls2(self.mlp(self.norm2(x)))

        return x

        
class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm_y = norm_layer(dim)

    def forward(self, x: Tensor, y: Tensor, pos=None) -> Tensor:
        
        x = x + self.ls1(self.attn(self.norm1(x), pos=pos[0]))
        y_ = self.norm_y(y)
        x = x + self.ls1(self.cross_attn(self.norm2(x), y_, pos=pos))
        x = x + self.ls2(self.mlp(self.norm3(x)))

        return x