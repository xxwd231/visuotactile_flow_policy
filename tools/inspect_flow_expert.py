"""Inspect the configured Action Expert; optionally run one CUDA bf16 smoke check."""

import argparse
from pathlib import Path
import statistics
import sys
import time

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visuotactile_flow.models import FlowActionExpert  # noqa: E402


def count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def parameter_breakdown(model: FlowActionExpert) -> dict[str, int]:
    return {
        "action input projection": count(model.action_input_projection),
        "action position embedding": model.action_position_embedding.numel(),
        "time embedder": count(model.timestep_embedder),
        "self-attention total": sum(count(b.self_attention) for b in model.blocks),
        "cross-attention total": sum(count(b.cross_attention) for b in model.blocks if b.cross_attention is not None),
        "FFN total": sum(count(b.ffn) for b in model.blocks),
        "AdaLN modulation total": sum(count(b.modulation) for b in model.blocks),
        "cross-attention query LayerNorm": sum(count(b.norm_ca) for b in model.blocks if b.norm_ca is not None),
        "final layer": count(model.final_modulation) + count(model.final_output),
    }


def smoke(model: FlowActionExpert) -> None:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        print("CUDA bf16 smoke: unavailable")
        return
    model = model.cuda().eval()
    device = torch.device("cuda")
    noisy = torch.randn(1, model.action_horizon, model.action_dim, device=device)
    flow_time = torch.tensor([0.5], device=device)
    condition = torch.randn(1, 15, model.hidden_dim, device=device)
    mask = torch.ones(1, 15, dtype=torch.bool, device=device)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(noisy, flow_time, condition, mask)  # warm-up
        torch.cuda.synchronize()
        durations = []
        for _ in range(3):
            start = time.perf_counter()
            output = model(noisy, flow_time, condition, mask)
            torch.cuda.synchronize()
            durations.append((time.perf_counter() - start) * 1000)
    if not torch.isfinite(output).all() or torch.count_nonzero(output):
        raise RuntimeError("Fresh zero-initialized expert must return finite zero velocity")
    print("CUDA bf16 output shape:", list(output.shape))
    print("CUDA allocated MiB:", round(torch.cuda.memory_allocated() / 2**20, 2))
    print("CUDA peak allocated MiB:", round(torch.cuda.max_memory_allocated() / 2**20, 2))
    print("CUDA median forward ms:", round(statistics.median(durations), 2))
    print("CUDA forward ms:", [round(d, 2) for d in durations])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run warm-up and three full-model CUDA bf16 forwards")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/model/flow_300m.yaml").read_text())
    head_dim = config.pop("head_dim")
    if config["hidden_dim"] // config["num_heads"] != head_dim:
        raise ValueError("Configured head_dim disagrees with hidden_dim / num_heads")
    model = FlowActionExpert(**config)
    breakdown = parameter_breakdown(model)
    assert sum(breakdown.values()) == count(model), "Parameter breakdown is incomplete"
    print("FlowActionExpert total params:", count(model))
    print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    for name, value in breakdown.items():
        print(f"{name}: {value}")
    print("Number of blocks:", len(model.blocks))
    print("Cross-attention block indices:", list(model.cross_attention_block_indices))
    if args.smoke:
        smoke(model)


if __name__ == "__main__":
    main()
