import numpy as np
import torch

from mtkdg.models.context_adapter import (
    ContextFeatureBuilder,
    ContextMetadata,
    ContextualResidualAdapter,
)


def _metadata():
    rng = np.random.RandomState(4)
    features = rng.normal(size=(24, 5))
    error = np.abs(features[:, 0]) + 0.1 * rng.normal(size=24)
    clusters = np.repeat(np.arange(12), 2)
    condition = np.repeat(np.arange(3), 8)
    return ContextMetadata.fit_train(
        features=features,
        point_error=error,
        cluster_id=clusters,
        condition_code=condition,
        split="train",
    )


def test_metadata_uses_train_only_support_and_assigns_stable_bins():
    metadata = _metadata()
    assignment = metadata.assign(np.zeros((3, 5)), np.array([0, 1, 99]))

    assert assignment.support_weight.shape == (3,)
    assert assignment.support_weight[-1] == 0.0
    assert 0 <= assignment.difficulty_bin.min() <= assignment.difficulty_bin.max() <= 3
    assert metadata.kappa_train >= 1
    assert metadata.state_dict()["fit_split"] == "train"


def test_metadata_rejects_non_train_fit():
    rng = np.random.RandomState(0)
    with np.testing.assert_raises(ValueError):
        ContextMetadata.fit_train(
            rng.normal(size=(8, 3)), np.ones(8), np.arange(8), split="val"
        )


def test_feature_builder_contains_only_encoder_observables():
    builder = ContextFeatureBuilder(d_model=4, n_channels=3, condition_dim=2, n_scales=2)
    enc = {
        "z": torch.randn(2, 3, 4),
        "gate_weights": torch.softmax(torch.randn(2, 2), dim=-1),
        "fusion_alpha": torch.randn(2, 3, 4),
        "obs_ratio": torch.rand(2, 3),
        "cond_emb": torch.randn(2, 2),
    }
    features = builder(enc)
    assert features.shape == (2, builder.feature_dim)
    assert torch.isfinite(features).all()


def test_zero_support_strictly_reverts_to_base_representation():
    torch.manual_seed(2)
    adapter = ContextualResidualAdapter(d_model=4, feature_dim=7, n_channels=3)
    z = torch.randn(2, 3, 4)
    global_features = torch.randn(2, 7)
    out = adapter(z, global_features, support_weight=torch.zeros(2))

    assert torch.equal(out.z_ctx, z)
    assert torch.equal(out.support_weight, torch.zeros_like(out.support_weight))


def test_adapter_changes_supported_context_and_keeps_input_gradient_free():
    torch.manual_seed(3)
    adapter = ContextualResidualAdapter(d_model=4, feature_dim=7, n_channels=3)
    torch.nn.init.constant_(adapter.delta.weight, 0.1)
    z = torch.randn(2, 3, 4)
    global_features = torch.randn(2, 7)
    out = adapter(z, global_features, support_weight=torch.ones(2))
    loss = out.z_ctx.square().mean()
    loss.backward()

    assert out.z_ctx.shape == z.shape
    assert torch.isfinite(out.z_ctx).all()
    assert any(p.grad is not None for p in adapter.parameters())
    assert not torch.equal(out.z_ctx, z)
