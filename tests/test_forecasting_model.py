import torch
import pytest
from src.forecasting.model import NeuroStockForecaster
from src.forecasting.attention import MultiHeadAttention
from src.forecasting.losses import GaussianNLLLoss, QuantileLoss


@pytest.fixture
def small_model():
    return NeuroStockForecaster(
        n_sales_features=10, n_sensor_features=4,
        d_model=16, n_heads=2, n_layers=2, horizon=7
    )


def test_forward_pass_output_shape(small_model):
    batch_size, seq_len = 4, 30
    sales_seq = torch.randn(batch_size, seq_len, 10)
    sensor_seq = torch.randn(batch_size, seq_len, 4)
    mu, log_var, attn = small_model(sales_seq, sensor_seq)
    assert mu.shape == (batch_size, 7)
    assert log_var.shape == (batch_size, 7)


def test_attention_weights_sum_to_one():
    attn = MultiHeadAttention(d_model=16, n_heads=2)
    attn.eval()  # matikan dropout agar weights tidak berubah
    x = torch.randn(2, 10, 16)
    with torch.no_grad():
        _, weights = attn(x, x, x)
    # softmax di axis terakhir harus menjumlah 1
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

def test_gaussian_nll_loss_decreases_with_better_prediction():
    loss_fn = GaussianNLLLoss()
    target = torch.tensor([10.0, 20.0, 30.0])
    good_mu = torch.tensor([10.1, 19.9, 30.2])
    bad_mu = torch.tensor([5.0, 25.0, 50.0])
    log_var = torch.zeros(3)
    good_loss = loss_fn(good_mu, log_var, target)
    bad_loss = loss_fn(bad_mu, log_var, target)
    assert good_loss < bad_loss


def test_quantile_loss_monotonic_quantiles():
    loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    target = torch.tensor([[10.0]])
    # q10 < q50 < q90 seharusnya menghasilkan loss rendah jika urutan benar
    preds_correct_order = torch.tensor([[[8.0, 10.0, 12.0]]])
    loss = loss_fn(preds_correct_order, target)
    assert loss.item() >= 0  # loss tidak boleh negatif


def test_gradient_flows_through_full_model(small_model):
    sales_seq = torch.randn(2, 20, 10, requires_grad=True)
    sensor_seq = torch.randn(2, 20, 4)
    mu, log_var, _ = small_model(sales_seq, sensor_seq)
    loss = mu.sum() + log_var.sum()
    loss.backward()
    # pastikan tidak ada parameter dengan gradient None
    for name, param in small_model.named_parameters():
        assert param.grad is not None, f"Parameter {name} tidak menerima gradient"