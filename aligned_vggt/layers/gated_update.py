import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedUpdate(nn.Module):
    def __init__(self, token_dim, num_tokens, init_gate=0.5):
        super().__init__()
        self.token_dim = token_dim
        self.num_tokens = num_tokens

        """
        self.new_ln = nn.LayerNorm(token_dim)
        self.ln = nn.LayerNorm(token_dim)
        self.token_mixing_ln = nn.LayerNorm(token_dim)
    

        # Per-global-token projection for the new token
        self.per_token_proj = nn.ModuleList([
            nn.Linear(token_dim, token_dim) for _ in range(num_tokens)
        ])

        # Linear layer to compute gate from concatenated old and new token
        self.gate_fc = nn.Linear(2 * token_dim, token_dim)
        nn.init.constant_(self.gate_fc.bias, 1.0) # Initialize gate bias to favor retaining previous state at first

        
        # Allow inter-token exchange through linear projection and residuals
        self.token_mixing_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, token_dim)
        )
        """

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

        bias_val = torch.log(torch.tensor(init_gate) / (1 - init_gate))  # sigmoid^-1(init_gate)
        nn.init.constant_(self.gate_mlp[-1].bias, bias_val)
        nn.init.normal_(self.gate_mlp[-1].weight, mean=0.0, std=0.1)  # small weights
        

    def forward(self, prev, new):
        """
        prev: [batch, num_tokens, token_dim] - previous token
        new:  [batch, 1, token_dim] - new token
        """

        B,N,D = prev.shape
        assert N == self.num_tokens and D == self.token_dim

        #computeCosineSimilarity(prev)

        new_scale = new.norm(dim=-1, keepdim=True)  # [B,1]

        memory_mean = prev.mean(dim=1, keepdim=True).expand_as(prev) # [batch, num_tokens, token_dim] #
        new_exp = new.expand_as(prev)
        #computer per token deltas
        delta_inputs = torch.cat([new_exp, prev * new_scale, memory_mean * new_scale], dim=-1) # [batch, num_tokens, 3*token_dim]
        deltas = torch.stack([self.delta_mlps[i](delta_inputs[:,i]) for i in range(self.num_tokens)],dim=1) # [batch, num_tokens, token_dim]
        
        #compute diff
        delta_diff = deltas - prev

        #compute gates
        gate_input = torch.cat([delta_diff, prev * new_scale], dim=-1).detach()  # [B,num_tokens, 2*D]
        gate = torch.sigmoid(self.gate_mlp(gate_input))  # [B,num_tokens,1]

        #orthogonalize
        delta_orth = delta_diff - (delta_diff * prev).sum(-1, keepdim=True) * prev #no need to divide by norm of memory dir since it is normalized

        #normalize
        delta_dir = F.normalize(delta_orth,dim=-1)

        #apply update
        new_memory_dir = prev + gate * delta_dir

        #normalize to direction
        new_memory_dir = F.normalize(new_memory_dir, dim=-1)  # keep unit norm
        
        """
        #normalize new tokens (prev are already normalized as output is normalized)
        new = self.new_ln(new)

        # Per-token projection: produce a slightly different new token per memory slot
        new_proj_list = [proj(new) for proj in self.per_token_proj]
        new_proj = torch.cat(new_proj_list, dim=1)  # [B, N, D]

        # Compute gate values between 0 and 1
        gate_input = torch.cat([prev, new_proj], dim=-1) # [B, N, 2D]
        gate = torch.sigmoid(self.gate_fc(gate_input))  # [B, N, D]
        
        # Gated residual update
        updated = gate * prev + (1.0 - gate) * new_proj

        # allow interaction between tokens
        mixed = self.token_mixing_ln(self.token_mixing_mlp(updated))
        updated = updated + mixed

        #normalize output
        updated = self.ln(updated)
        """

        #if not self.training:
            #print(f"Mean gate per chunk: {gate.mean()}")

        return new_memory_dir
    

def computeCosineSimilarity(tokens):
    tokens_norm = F.normalize(tokens, p=2, dim=-1, eps=1e-8)  # B x N x D
    B, N = tokens.shape[:2]

    cosine_sim = torch.bmm(tokens_norm, tokens_norm.transpose(1, 2))  # B x N x N
    mask = 1.0 - torch.eye(N, device=tokens_norm.device).unsqueeze(0)
    cosine_sim_offdiag = cosine_sim * mask
    avg_similarity = (cosine_sim_offdiag.sum() / (B * N * (N - 1))).item()
    print(f"Average token similarity: {avg_similarity}")