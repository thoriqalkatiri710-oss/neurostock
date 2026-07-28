import torch
from torch import nn

from .cross_modal import CrossModalFusion
from .positional_encoding import PositionalEncoding


class NeuroStockForecaster(nn.Module):
    """
    Encoder-only Transformer dengan cross-modal attention untuk fusi
    data sales dan sensor IoT simulasi.

    Output: mu dan log_var per timestep untuk uncertainty quantification.
    """

    def __init__(self, n_sales_features: int, n_sensor_features: int,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 3,
                 dropout: float = 0.1, horizon: int = 14):
        super().__init__()
        self.sales_embed = nn.Linear(n_sales_features, d_model)
        self.sensor_embed = nn.Linear(n_sensor_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cross_modal = CrossModalFusion(d_model, n_heads, dropout)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2)  # output: mu dan log_var
        )
        self.horizon = horizon

    def forward(self, sales_seq: torch.Tensor, sensor_seq: torch.Tensor) -> tuple:
        sales_emb = self.pos_encoding(self.sales_embed(sales_seq))
        sensor_emb = self.pos_encoding(self.sensor_embed(sensor_seq))
        encoded = self.encoder(sales_emb)
        fused, attn_weights = self.cross_modal(encoded, sensor_emb)

        # Ambil horizon timestep terakhir untuk multi-step forecasting
        last_repr = fused[:, -self.horizon:, :]
        output = self.output_head(last_repr)
        mu, log_var = output[..., 0], output[..., 1]
        return mu, log_var, attn_weights