import torch

from mtkdg.models.prob_heads import (
    QUANTILE_LEVELS,
    ProbabilisticOutput,
    StateQuantileHead,
    RULQuantileHead,
    pinball_loss,
)


def test_rul_quantile_head_returns_nonnegative_monotone_quantiles():
    head = RULQuantileHead(d_model=8)
    out = head(torch.randn(5, 17, 8), torch.randn(5, 8))

    assert out.shape == (5, len(QUANTILE_LEVELS))
    assert (out >= 0).all()
    assert (out[..., 1:] >= out[..., :-1]).all()


def test_state_quantile_head_has_expected_contract():
    head = StateQuantileHead(d_model=8, n_channels=4, pred_len=6)
    out = head(torch.randn(3, 4, 8))

    assert out.shape == (3, 6, 4, len(QUANTILE_LEVELS))
    assert (out[..., 1:] >= out[..., :-1]).all()


def test_quantile_head_supports_pinball_gradient_and_output_record():
    head = RULQuantileHead(d_model=8)
    prediction = head(torch.randn(4, 17, 8), torch.randn(4, 8))
    target = torch.rand(4) * 150
    loss = pinball_loss(prediction, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert head.net[-1].weight.grad is not None
    output = ProbabilisticOutput("quantile", {}, prediction, "raw cycle", "rul_quantile_v2")
    assert output.quantiles is prediction
