"""Local-only four-branch parity check against the trusted old tactile DP.

This tool is the only code here that imports the legacy vendor package.
Runtime code under src/visuotactile_flow never does so.
"""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from visuotactile_flow.encoders import (  # noqa: E402
    DepthEncoding, TactileEncoder, VisionEncoder, load_encoder_weights,
    raw_depth_to_policy_depth,
)

DEFAULT_LEGACY_VENDOR = Path(
    "/home/pine/openpi/ros2_teleop_dataset/tactile_RL/offline_RL/usb_insertion/vendor"
)


def compare(checkpoint: Path, weights_dir: Path, legacy_vendor: Path) -> dict:
    if not legacy_vendor.is_dir():
        raise FileNotFoundError(legacy_vendor)
    sys.path.insert(0, str(legacy_vendor))
    import hydra  # noqa: PLC0415

    torch.set_num_threads(2)
    torch.manual_seed(17)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    legacy = hydra.utils.instantiate(payload["policy_config"]["obs_encoder"])
    prefix = "obs_encoder."
    old_state = {key.removeprefix(prefix): value for key, value in
                 payload["state_dicts"]["model"].items() if key.startswith(prefix)}
    legacy.load_state_dict(old_state, strict=True)
    legacy.eval()

    vision = VisionEncoder(freeze=True).eval()
    tactile = TactileEncoder(depth_encoding=DepthEncoding.POLICY_NORMALIZED, freeze=True).eval()
    for name in ("external_rgb", "wrist_rgb"):
        load_encoder_weights(getattr(vision, name), weights_dir / f"{name}.pt")
    for name in ("tactile_depth_0", "tactile_depth_1"):
        load_encoder_weights(getattr(tactile, name), weights_dir / f"{name}.pt")

    # RGB resembles the policy-facing 320x240 RGB uint8 after robot_io resize.
    external = torch.randint(0, 256, (1, 3, 240, 320), dtype=torch.uint8)
    wrist = torch.randint(0, 256, (1, 3, 240, 320), dtype=torch.uint8)
    # SDK-like depth is explicitly converted once, as in the old worker.
    raw0 = torch.randn(1, 1, 288, 384) * 0.2
    raw1 = torch.randn(1, 1, 288, 384) * 0.2
    depth0 = raw_depth_to_policy_depth(raw0, input_encoding=DepthEncoding.RAW_SDK_DEPTH)
    depth1 = raw_depth_to_policy_depth(raw1, input_encoding=DepthEncoding.RAW_SDK_DEPTH)
    input_frames = {"external_rgb": external, "wrist_rgb": wrist,
                    "tactile_depth_0": depth0, "tactile_depth_1": depth1}
    with torch.inference_mode():
        # Use the old full encoder forward: its documented branch order is
        # sorted(rgb_keys), followed by the seven state values.
        old_combined = legacy({**input_frames, "agent_pos": torch.zeros(1, 7)}, deterministic=True)
        new_outputs = {
            **vision(external_rgb=external[:, None], wrist_rgb=wrist[:, None]),
            **tactile(tactile_depth_0=depth0[:, None], tactile_depth_1=depth1[:, None]),
        }
        reports = {}
        for index, name in enumerate(legacy.rgb_keys):
            old_feature = old_combined[:, index * 512:(index + 1) * 512]
            new_feature = new_outputs[name][:, 0]
            old_prepared = legacy._apply_transform(name, input_frames[name], deterministic=True)
            new_prepared = (vision.preprocess(input_frames[name][:, None])[:, 0]
                            if name.endswith("rgb") else tactile.preprocess(input_frames[name][:, None])[:, 0])
            feature_error = (old_feature - new_feature).abs()
            reports[name] = {
                "shape": list(new_feature.shape),
                "max_abs_error": float(feature_error.max()),
                "mean_abs_error": float(feature_error.mean()),
                "allclose": bool(torch.allclose(old_feature, new_feature, atol=1e-5, rtol=1e-4)),
                "preprocess_max_abs_error": float((old_prepared - new_prepared).abs().max()),
            }
    return {"device": "cpu", "dtype": "float32", "boundary":
            "policy-facing RGB uint8 320x240 and SDK-like depth normalized once; includes old and new preprocessing",
            "atol": 1e-5, "rtol": 1e-4, "branches": reports}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "artifacts/legacy_encoders")
    parser.add_argument("--legacy-vendor", type=Path, default=DEFAULT_LEGACY_VENDOR)
    args = parser.parse_args()
    report = compare(args.checkpoint, args.weights_dir, args.legacy_vendor)
    print(json.dumps(report, indent=2))
    if not all(branch["allclose"] for branch in report["branches"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
