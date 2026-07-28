import torch
from torch import nn

from .attention import MultiHeadAttention


class CrossModalFusion(nn.Module):
    """
    Fusi sales embedding dan sensor embedding via cross-modal attention.

    - Query: sales_emb (modalitas utama)
    - Key/Value: sensor_emb (modalitas pendukung)
    - Gating mechanism: model belajar seberapa besar bobot sensor
      per timestep — bisa "mematikan" pengaruh sensor saat tidak relevan.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, sales_emb: torch.Tensor, sensor_emb: torch.Tensor):
        # sales_emb sebagai Query, sensor_emb sebagai Key & Value
        fused, attn_weights = self.cross_attn(
            query=sales_emb,
            key=sensor_emb,
            value=sensor_emb
        )
        fused = self.norm(fused + sales_emb)  # residual connection

        # Gating: seberapa besar pengaruh sensor vs sales murni per timestep
        gate_input = torch.cat([sales_emb, fused], dim=-1)
        gate_value = self.gate(gate_input)
        output = gate_value * fused + (1 - gate_value) * sales_emb

        return output, attn_weights