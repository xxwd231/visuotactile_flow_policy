"""Export four portable branches from one trusted local tactile DP checkpoint.

This is an offline migration tool. The installed package never imports the old
repository or loads its full DP checkpoint at runtime.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from visuotactile_flow.encoders import ResNet18Encoder, load_encoder_weights  # noqa: E402


BRANCHES = {
    "external_rgb": 3,
    "wrist_rgb": 3,
    "tactile_depth_0": 1,
    "tactile_depth_1": 1,
}


def export(checkpoint: Path, output_dir: Path) -> dict:
    checkpoint = checkpoint.resolve(strict=True)
    # The local old DP checkpoint contains a Hydra config; trusted input only.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["policy_config"]["obs_encoder"]
    if config["rgb_model"]["name"] != "resnet18" or config["share_rgb_model"] or not config["use_group_norm"]:
        raise ValueError("Checkpoint does not match independent GroupNorm ResNet18 branches")
    if set(BRANCHES) - set(config["shape_meta"]["obs"]):
        raise ValueError("Checkpoint has missing tactile or RGB observations")
    all_weights = payload["state_dicts"]["model"]
    exports: dict[str, dict] = {}
    extracted: dict[str, dict] = {}
    for name, channels in BRANCHES.items():
        prefix = f"obs_encoder.key_model_map.{name}."
        state = {key.removeprefix(prefix): value.detach().cpu()
                 for key, value in all_weights.items() if key.startswith(prefix)}
        if not state:
            raise ValueError(f"No weights found for {name}")
        branch = ResNet18Encoder(channels)
        branch.backbone.load_state_dict(state, strict=True)
        extracted[name] = {"format_version": 1, "encoder_name": name,
                           "in_channels": channels, "output_dim": 512,
                           "state_dict": state}
        exports[name] = {"file": f"{name}.pt", "in_channels": channels,
                         "output_dim": 512, "parameter_count": sum(p.numel() for p in branch.parameters()),
                         "source_key_prefix": prefix}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, branch_payload in extracted.items():
        path = output_dir / f"{name}.pt"
        temp = path.with_suffix(".pt.tmp")
        torch.save(branch_payload, temp)
        temp.replace(path)
        # Verify the portable artifact through the runtime loader.
        load_encoder_weights(ResNet18Encoder(BRANCHES[name]), path)
        exports[name]["size_bytes"] = path.stat().st_size
    metadata = {
        "format_version": 1,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_size": checkpoint.stat().st_size,
        "export_time": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": str(payload.get("upstream_commit")),
        "architecture": "torchvision.models.resnet18; fc=Identity; independent branches",
        "normalization_type": "GroupNorm(num_groups=num_channels//16, eps=1e-5, affine=True)",
        "preprocessing_contract": {
            "rgb": "RGB uint8 -> /255 -> Resize(224,224) -> CenterCrop(216,216) -> ImageNet mean/std",
            "tactile_depth": "raw SDK depth -> clip(depth/0.7,-1,1) OR already policy-normalized; Resize(224,224)",
        },
        "source_key_prefixes": {name: entry["source_key_prefix"] for name, entry in exports.items()},
        "encoders": exports,
    }
    temp = output_dir / "metadata.json.tmp"
    temp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(output_dir / "metadata.json")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/legacy_encoders")
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
