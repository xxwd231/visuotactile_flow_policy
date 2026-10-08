import json
from pathlib import Path

import pytest
import torch
import yaml

from visuotactile_flow.data.schema import DataConfig
from visuotactile_flow.normalization import (
    FixedRangeNormalizer, IdentityNormalizer, MeanStdNormalizer,
    MinMaxNormalizerV2, QuantileNormalizer, StructuredActionNormalizer,
    StructuredStateNormalizer, analyze_action_distribution,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("factory", [IdentityNormalizer, MeanStdNormalizer,
                                      MinMaxNormalizerV2, QuantileNormalizer])
def test_base_round_trip_and_global_feature_stats(factory):
    x = torch.randn(20, 16, 10, generator=torch.Generator().manual_seed(8))
    normalizer = factory().fit(x)
    assert normalizer.scale.shape == (10,)
    y = normalizer.normalize(x)
    torch.testing.assert_close(normalizer.denormalize(y), x, atol=1e-5, rtol=1e-5)
    loaded = factory().load_state_dict(json.loads(json.dumps(normalizer.state_dict())))
    torch.testing.assert_close(loaded.normalize(x), y)


def test_identity_and_fixed_range_physical_gripper():
    values = torch.tensor([[0.0], [0.5], [1.0]])
    identity = IdentityNormalizer().fit(values)
    torch.testing.assert_close(identity.normalize(values), values)
    normalizer = FixedRangeNormalizer().fit(values)
    torch.testing.assert_close(normalizer.normalize(values), torch.tensor([[-1.0], [0.0], [1.0]]))
    torch.testing.assert_close(normalizer.denormalize(normalizer.normalize(values)), values)
    with pytest.raises(ValueError, match="physical range"):
        FixedRangeNormalizer().fit(torch.tensor([[0.0], [1.1]]))


def test_near_constant_protection_and_floor_flags():
    x = torch.tensor([[1.0, 2.0], [1.0 + 1e-6, 3.0]], dtype=torch.float64)
    for normalizer in (MeanStdNormalizer(), MinMaxNormalizerV2(), QuantileNormalizer()):
        normalizer.fit(x)
        assert normalizer.state_dict()["floored_channels"] == [True, False]
        assert float(normalizer.scale[0]) <= 20000
        assert "stats" in normalizer.state_dict()
    mm = MinMaxNormalizerV2().fit(x)
    torch.testing.assert_close(mm.normalize(x)[:, 0], torch.zeros(2, dtype=x.dtype))
    torch.testing.assert_close(mm.denormalize(mm.normalize(x))[:, 0],
                               mm.stats["center"][0].expand(2))
    ms = MeanStdNormalizer().fit(x)
    torch.testing.assert_close(ms.denormalize(ms.normalize(x)), x)
    assert ms.stats["effective_std"][0] == pytest.approx(1e-4)
    q = QuantileNormalizer().fit(x)
    assert q.stats["effective_span"][0] == pytest.approx(1e-4)


def test_quantile_ood_is_not_clipped():
    x = torch.arange(100, dtype=torch.float32)[:, None]
    normalizer = QuantileNormalizer().fit(x)
    assert normalizer.normalize(torch.tensor([[-100.0], [200.0]])).min() < -1
    assert normalizer.normalize(torch.tensor([[-100.0], [200.0]])).max() > 1


def test_minmax_ood_is_not_clipped():
    normalizer = MinMaxNormalizerV2().fit(torch.tensor([[0.0], [1.0]]))
    torch.testing.assert_close(normalizer.normalize(torch.tensor([[-1.0], [2.0]])),
                               torch.tensor([[-3.0], [3.0]]))


def _config(name):
    return yaml.safe_load((ROOT / "configs/normalization" / name).read_text())


def test_action_sections_modes_and_metadata_round_trip():
    modes = _config("action_robust_candidate.yaml")
    x = torch.randn(12, 16, 10) * 0.1
    x[..., 9] = 0.7
    config = DataConfig(tactile_representation_source="legacy_video_reconstructed")
    normalizer = StructuredActionNormalizer(modes, data_config=config, dataset_fingerprint="sha256:example").fit(x)
    y = normalizer.normalize(x)
    assert y.shape == x.shape
    torch.testing.assert_close(y[..., 3:9], x[..., 3:9])
    torch.testing.assert_close(y[..., 9], torch.full_like(y[..., 9], 0.4))
    torch.testing.assert_close(normalizer.denormalize(y), x, atol=1e-5, rtol=1e-5)
    saved = json.loads(json.dumps(normalizer.state_dict()))
    assert saved["metadata"]["tactile_representation_source"] == "legacy_video_reconstructed"
    assert saved["metadata"]["target_start_offset_steps"] == 1
    assert saved["metadata"]["fit_split"] == "train"
    restored = StructuredActionNormalizer(modes).load_state_dict(saved)
    torch.testing.assert_close(restored.normalize(x), y)
    saved["metadata"]["action_label_source"] = "commanded_target"
    with pytest.raises(ValueError):
        StructuredActionNormalizer(modes).load_state_dict(saved)


def test_state_sections_and_independent_stats():
    action = torch.randn(8, 16, 10)
    action[..., 9] = 0.6
    state = torch.randn(8, 3, 7) + 5
    state[..., 6] = 0.6
    a = StructuredActionNormalizer(_config("action_meanstd_candidate.yaml")).fit(action)
    s = StructuredStateNormalizer(_config("state_meanstd_candidate.yaml")).fit(state)
    assert s.normalize(state).shape == state.shape
    assert a.parts["translation"].stats["mean"].shape == (3,)
    assert s.parts["joints"].stats["mean"].shape == (6,)
    assert not torch.allclose(a.parts["translation"].stats["mean"], s.parts["joints"].stats["mean"][:3])
    torch.testing.assert_close(s.denormalize(s.normalize(state)), state, atol=1e-5, rtol=1e-5)


def test_all_candidate_configs_construct():
    for name in ("action_legacy_minmax.yaml", "action_meanstd_candidate.yaml",
                 "action_robust_candidate.yaml"):
        StructuredActionNormalizer(_config(name))
    for name in ("state_legacy_minmax.yaml", "state_meanstd_candidate.yaml"):
        StructuredStateNormalizer(_config(name))


def test_diagnostics_horizon_flow_scale_and_tail_fractions():
    x = torch.randn(10, 16, 10) * 0.1
    x[..., 9] = 0.5
    normalizer = StructuredActionNormalizer(_config("action_robust_candidate.yaml")).fit(x)
    report = analyze_action_distribution(x, normalizer, generator=torch.Generator().manual_seed(4))
    for key in ("raw", "normalized"):
        assert set(report[key]) == {"min", "max", "mean", "std", "q01", "q99"}
        assert all(value.shape == (10,) for value in report[key].values())
    for key in ("per_horizon_mean", "per_horizon_std", "per_horizon_q01", "per_horizon_q99"):
        assert report[key].shape == (16, 10)
    for key in ("fraction_abs_gt_1", "fraction_abs_gt_2", "fraction_abs_gt_3",
                "action_rms_per_channel", "normalized_action_std_per_channel",
                "std_ratio", "velocity_mse_mc_per_channel", "velocity_mse_expected_per_channel"):
        assert report[key].shape == (10,)
        assert torch.isfinite(report[key]).all()
    assert report["source_std"] == 1.0


def test_v1_legacy_state_is_rejected_by_v2():
    with pytest.raises(ValueError, match="version"):
        MinMaxNormalizerV2().load_state_dict({"type": "minmax", "version": 1})
