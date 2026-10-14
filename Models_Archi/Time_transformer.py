import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchview import draw_graph
from torchviz import make_dot
from typing import List


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation   # left-only causal padding
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
 
    def forward(self, x):
        # x: (B, C, L)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TimesformerLayer(nn.Module):

 
    def __init__(
        self,
        d_model: int,
        filters: int,
        k_size: int,
        dilation: int,
        head_size: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout_p = dropout
        # embed_dim of the Transformer branch = head_size * num_heads
        # (In TF, key_dim is per-head; total projection = key_dim * num_heads)
        embed_dim = head_size * num_heads
 
        # ── TCN branch ────────────────────────────────────────────────────
        self.causal_conv = CausalConv1d(d_model, filters, k_size, dilation)
        self.ln_tcn      = nn.LayerNorm(filters, eps=1e-6)
        self.spatial_drop = nn.Dropout(dropout)
 
        # ── Transformer branch (self-attention + FF) ──────────────────────
        # Project input to embed_dim if d_model != embed_dim
        self.proj_trans_in = (
            nn.Linear(d_model, embed_dim)
            if d_model != embed_dim
            else nn.Identity()
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_sa = nn.Dropout(dropout)
        self.ln_sa   = nn.LayerNorm(embed_dim, eps=1e-6)
 
        # Feed-forward: Conv1d(filters, k=1) → ReLU → Dropout
        #               → Conv1d(embed_dim, k=1)  [project back]
        # (mirrors: Conv1D(filters,k=1) then Conv1D(filters=trans_inputs.shape[-1], k=1))
        self.ff_conv1 = nn.Conv1d(embed_dim, filters,   kernel_size=1)
        self.drop_ff  = nn.Dropout(dropout)
        self.ff_conv2 = nn.Conv1d(filters,   embed_dim, kernel_size=1)
        self.ln_ff    = nn.LayerNorm(embed_dim, eps=1e-6)
 
        # ── Bidirectional Cross-Attention ─────────────────────────────────
        # Direction A: Q=trans_out (embed_dim), K=V=tcn_out (filters)
        # Direction B: Q=tcn_out   (filters),   K=V=trans_out (embed_dim)
        # Official code uses num_heads=1 for both cross-attention calls.
        self.cross_attn_A = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
            kdim=filters,
            vdim=filters,
        )
        self.cross_attn_B = nn.MultiheadAttention(
            embed_dim=filters,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
            kdim=embed_dim,
            vdim=embed_dim,
        )
        self.drop_ca_A = nn.Dropout(dropout)
        self.drop_ca_B = nn.Dropout(dropout)
        self.ln_ca_A   = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln_ca_B   = nn.LayerNorm(filters,   eps=1e-6)
 
    def forward(self, tcn_inputs: torch.Tensor, trans_inputs: torch.Tensor):
        """
        Args
        ----
        tcn_inputs   : (B, L, d_model)
        trans_inputs : (B, L, d_model)
 
        Returns
        -------
        chnl_tcn   : (B, L, filters)
        chnl_trans : (B, L, embed_dim)
        """
 
        # ── TCN branch ────────────────────────────────────────────────────
        # Conv1d works on (B, C, L)
        x = tcn_inputs.transpose(1, 2)           # (B, d_model, L)
        x = F.relu(self.causal_conv(x))
        x = x.transpose(1, 2)                    # (B, L, filters)
        x = self.ln_tcn(x)
        tcn_out = self.spatial_drop(x)            # (B, L, filters)
 
        # ── Transformer branch: self-attention ────────────────────────────
        t = self.proj_trans_in(trans_inputs)      # (B, L, embed_dim)
        x, _ = self.self_attn(t, t, t)
        x = self.drop_sa(x)
        x = self.ln_sa(x)
        res = x + t                               # (B, L, embed_dim)
 
        # Feed-forward Conv1d x 2
        x = res.transpose(1, 2)                   # (B, embed_dim, L)
        x = F.relu(self.ff_conv1(x))
        x = self.drop_ff(x)
        x = self.ff_conv2(x)                      # (B, embed_dim, L)
        x = x.transpose(1, 2)                     # (B, L, embed_dim)
        x = self.ln_ff(x)
        trans_out = x + res                        # (B, L, embed_dim)
 
        # ── Bidirectional Cross-Attention ─────────────────────────────────
        # Direction A: Q=trans_out, K=V=tcn_out  → chnl_trans
        x, _ = self.cross_attn_A(
            query=trans_out, key=tcn_out, value=tcn_out
        )
        x = self.drop_ca_A(x)
        x = self.ln_ca_A(x)
        chnl_trans = x + trans_out                 # (B, L, embed_dim)
 
        # Direction B: Q=tcn_out, K=V=trans_out  → chnl_tcn
        x, _ = self.cross_attn_B(
            query=tcn_out, key=trans_out, value=trans_out
        )
        x = self.drop_ca_B(x)
        x = self.ln_ca_B(x)
        chnl_tcn = x + tcn_out                     # (B, L, filters)
 
        return chnl_tcn, chnl_trans
 
 

class TimeTransformerEncoder(nn.Module):
 
    def __init__(
        self,
        in_features : int,
        d_model     : int       = 32,
        head_size   : int       = 16,
        num_heads   : int       = 2,
        dilations   : list      = [1],
        k_size      : int       = 4,
        dropout     : float     = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
 
        self.layers = nn.ModuleList([
            TimesformerLayer(
                d_model   = d_model,
                filters   = d_model,
                k_size    = k_size,
                dilation  = d,
                head_size = head_size,
                num_heads = num_heads,
                dropout   = dropout,
            )
            for d in dilations
        ])
 
        embed_dim = head_size * num_heads
        # output dim of this encoder after GlobalAvgPool + concat of both branches
        self.out_dim = d_model + embed_dim
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args    x : (B, L, in_features)
        Returns z : (B, out_dim)   pooled embedding
        """
        h = self.input_proj(x)          # (B, L, d_model)
 
        tcn_stream   = h
        trans_stream = h
 
        for layer in self.layers:
            tcn_stream, trans_stream = layer(tcn_stream, trans_stream)
 
        # concat TCN + Trans streams along feature axis, then avg-pool over time
        fused = torch.cat([tcn_stream, trans_stream], dim=-1)   # (B, L, out_dim)
        z     = fused.mean(dim=1)                                # (B, out_dim)
        return z
 
 
class MultiModalTimeTransformer(nn.Module):
    def __init__(
        self,
        branch_cfgs : list,           # list of dicts, one per modality
        hidden_dim  : int   = 32,
        num_classes : int   = 2,
        dropout     : float = 0.2,
    ):
        super().__init__()
 
        if len(branch_cfgs) < 1:
            raise ValueError("branch_cfgs must contain at least 1 branch config.")
 
        # Build one encoder per modality, stored in a ModuleList so PyTorch
        # registers all their parameters correctly.
        self.branches = nn.ModuleList([
            TimeTransformerEncoder(**cfg) for cfg in branch_cfgs
        ])
 
        # fused_dim = sum of every branch's output embedding size
        fused_dim = sum(b.out_dim for b in self.branches)
 
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
 
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) != len(self.branches):
            raise ValueError(
                f"Expected {len(self.branches)} input tensors, got {len(inputs)}."
            )
 
        # Encode each modality independently → list of (B, out_dim_i)
        embeddings = [branch(x) for branch, x in zip(self.branches, inputs)]
 
        # Late fusion: concatenate all embeddings → (B, fused_dim)
        z = torch.cat(embeddings, dim=-1)
 
        return self.classifier(z)                     # (B, num_classes)
 
    def encode(self, *inputs: torch.Tensor):
        """
        Return individual branch embeddings as a tuple — useful for
        visualisation, t-SNE, or branch-level analysis.
 
        Returns: tuple of (B, out_dim_i) tensors, one per branch.
        """
        return tuple(branch(x) for branch, x in zip(self.branches, inputs))


class DualBranchTimeTransformer(nn.Module):
    def __init__(
        self,
        branch_a_cfg : dict,
        branch_b_cfg : dict,
        hidden_dim   : int   = 64,
        num_classes  : int   = 1,
        dropout      : float = 0.2,
    ):
        super().__init__()
 
        self.branch_a = TimeTransformerEncoder(**branch_a_cfg)
        self.branch_b = TimeTransformerEncoder(**branch_b_cfg)
 
        fused_dim = self.branch_a.out_dim + self.branch_b.out_dim
 
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
 
    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:

        z_a = self.branch_a(x_a)                    # (B, out_dim_a)
        z_b = self.branch_b(x_b)                    # (B, out_dim_b)
        z   = torch.cat([z_a, z_b], dim=-1)         # (B, out_dim_a + out_dim_b)
        return self.classifier(z)                    # (B, num_classes)
 
    def encode(self, x_a: torch.Tensor, x_b: torch.Tensor):
        return self.branch_a(x_a), self.branch_b(x_b)

BRANCH_A = dict(
    in_features = 1,
    d_model     = 32,
    head_size   = 16,
    num_heads   = 2,        # embed_dim = 16*2 = 32
    dilations   = [1],
    k_size      = 4,
    dropout     = 0.2,
)

BRANCH_B = dict(
    in_features = 1,
    d_model     = 32,
    head_size   = 16,
    num_heads   = 2,
    dilations   = [1],
    k_size      = 4,
    dropout     = 0.2,
)

BRANCH_C = dict(
    in_features = 1,
    d_model     = 32,
    head_size   = 16,
    num_heads   = 2,
    dilations   = [1],
    k_size      = 4,
    dropout     = 0.2,
)


# model = DualBranchTimeTransformer(
# branch_a_cfg = BRANCH_A,
# branch_b_cfg = BRANCH_B,
# hidden_dim   = 64,
# num_classes  = 2,
# dropout      = 0.2,
# )
# model_graph.visual_graph.render(filename="DualBranchTimeTransformer", format="png")


model = MultiModalTimeTransformer( [BRANCH_A, BRANCH_B , BRANCH_C],
hidden_dim   = 32,
num_classes  = 2,
dropout      = 0.2,
)

input_a = torch.randn(1, 336, 1)   # (B, L_a, F_a)
input_b = torch.randn(1, 336, 1)  # (B, L_b, F_b)
input_c = torch.randn(1, 336, 1)  # (B, L_c, F_c)


model_graph = draw_graph(
    model,
    input_data=(input_a, input_b, input_c),  
    expand_nested=True
)


total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total: {total_params}")
print(f"Trainable: {trainable_params}")


# model_graph.visual_graph.render(filename="MultiModalTimeTransformer", format="png")
