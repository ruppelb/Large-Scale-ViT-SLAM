import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedUpdate(nn.Module):
    """
    Gated update module for memory tokens.
    Given a set of memory tokens and an update vector, computes per-token updates
    using per-token MLPs and gating mechanisms.
    Memory tokens are assumed to be normalized.
    Args:
        token_dim (int): Dimension of each memory token.
        num_tokens (int): Number of memory tokens.
        init_gate (float): Initial gate value between 0 and 1.
    """
    def __init__(self, token_dim: int, num_tokens: int, init_gate: float = 0.5):
        super().__init__()
        self.token_dim = token_dim
        self.num_tokens = num_tokens

        # Per-memory MLPs for producing memory-specific frame representations
        self.delta_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(token_dim*3, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim)
            )
            for _ in range(num_tokens)
        ])

        # Per-memory MLPs for per-token gate computation
        self.gate_mlp = nn.Sequential(
            nn.Linear(token_dim*2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 1)
        )

        bias_val = torch.log(torch.tensor(init_gate) / (1 - init_gate))  # inv_sigmoid(init_gate)
        nn.init.constant_(self.gate_mlp[-1].bias, bias_val)
        nn.init.normal_(self.gate_mlp[-1].weight, mean=0.0, std=0.1)  # small weights
        

    def forward(self, memory: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the gated update module.
        Args:
            memory (torch.Tensor): Memory tokens of shape (B, N, D).
            update (torch.Tensor): Update vector of shape (B, D).
        Returns:
            torch.Tensor: Updated memory tokens of shape (B, N, D).
        """
        B,N,D = memory.shape
        assert N == self.num_tokens and D == self.token_dim

        update_scale = update.norm(dim=-1, keepdim=True)  # (B, 1)
        update = update.expand_as(memory)

        memory_mean_scaled = memory.mean(dim=1, keepdim=True).expand_as(memory) * update_scale # (B, N, D)
        memory_scaled =  memory * update_scale
        
        # compute per token deltas
        delta_inputs = torch.cat([update, memory_scaled, memory_mean_scaled], dim=-1) # (B, N, 3*D)
        deltas = torch.stack([self.delta_mlps[i](delta_inputs[:,i]) for i in range(self.num_tokens)],dim=1) # (B, N, D)
        
        # compute diff
        delta_diff = deltas - memory

        # compute gates
        gate_input = torch.cat([delta_diff, memory_scaled], dim=-1).detach()  # (B, N, 2*D)
        gate = torch.sigmoid(self.gate_mlp(gate_input))  # (B, N, 1)

        # orthogonalize & normalize diff
        delta_orth = delta_diff - (delta_diff * memory).sum(-1, keepdim=True) * memory # no need to divide by norm of memory dir since it is normalized
        delta_dir = F.normalize(delta_orth,dim=-1)

        # apply update & normalize to direction
        new_memory_dir = F.normalize(memory + gate * delta_dir, dim=-1)

        return new_memory_dir