"""Lightweight projection of separate encoder features into condition tokens."""

import torch
from torch import nn

from .schema import ConditionOutput, MODALITY_ORDER, NUM_MODALITIES


class ConditionAdapter(nn.Module):
    def __init__(
        self,
        model_dim: int = 1024,
        encoder_feature_dim: int = 512,
        state_dim: int = 7,
        max_history: int = 8,
    ) -> None:
        super().__init__()
        if min(model_dim, encoder_feature_dim, state_dim, max_history) < 1:
            raise ValueError("All adapter dimensions must be positive")
        self.model_dim = model_dim
        self.encoder_feature_dim = encoder_feature_dim
        self.state_dim = state_dim
        self.max_history = max_history

        def projection() -> nn.Sequential:
            return nn.Sequential(nn.LayerNorm(encoder_feature_dim),
                                 nn.Linear(encoder_feature_dim, model_dim))

        self.external_rgb_projection = projection()
        self.wrist_rgb_projection = projection()
        self.tactile_depth_0_projection = projection()
        self.tactile_depth_1_projection = projection()
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, encoder_feature_dim), nn.SiLU(),
            nn.Linear(encoder_feature_dim, model_dim),
        )
        self.modality_embedding = nn.Embedding(NUM_MODALITIES, model_dim)
        self.temporal_embedding = nn.Embedding(max_history, model_dim)
        self.final_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        *,
        external_rgb: torch.Tensor,
        wrist_rgb: torch.Tensor,
        tactile_depth_0: torch.Tensor,
        tactile_depth_1: torch.Tensor,
        agent_pos: torch.Tensor,
        external_rgb_valid: torch.Tensor | None = None,
        wrist_rgb_valid: torch.Tensor | None = None,
        tactile_depth_0_valid: torch.Tensor | None = None,
        tactile_depth_1_valid: torch.Tensor | None = None,
        agent_pos_valid: torch.Tensor | None = None,
    ) -> ConditionOutput:
        features = (external_rgb, wrist_rgb, tactile_depth_0, tactile_depth_1, agent_pos)
        if external_rgb.ndim != 3:
            raise ValueError("external_rgb must have shape [B,T,encoder_feature_dim]")
        batch, history = external_rgb.shape[:2]
        if batch < 1 or history < 1 or history > self.max_history:
            raise ValueError(f"Expected B>0 and 1<=T<={self.max_history}; got B={batch}, T={history}")
        for name, feature in zip(MODALITY_ORDER, features):
            expected_dim = self.state_dim if name == "agent_pos" else self.encoder_feature_dim
            if feature.shape != (batch, history, expected_dim):
                raise ValueError(f"{name} must have shape [{batch},{history},{expected_dim}]")
            if feature.device != external_rgb.device or feature.dtype != external_rgb.dtype:
                raise ValueError("All features must share device and dtype")
            if not feature.is_floating_point():
                raise ValueError("Features must be floating point")

        masks = (external_rgb_valid, wrist_rgb_valid, tactile_depth_0_valid,
                 tactile_depth_1_valid, agent_pos_valid)
        checked_masks = []
        for name, mask in zip(MODALITY_ORDER, masks):
            if mask is None:
                mask = torch.ones((batch, history), device=external_rgb.device, dtype=torch.bool)
            elif mask.shape != (batch, history) or mask.dtype != torch.bool or mask.device != external_rgb.device:
                raise ValueError(f"{name}_valid must be bool [{batch},{history}] on the feature device")
            checked_masks.append(mask)

        projected = (
            self.external_rgb_projection(external_rgb),
            self.wrist_rgb_projection(wrist_rgb),
            self.tactile_depth_0_projection(tactile_depth_0),
            self.tactile_depth_1_projection(tactile_depth_1),
            self.state_mlp(agent_pos),
        )
        # [B,T,5,D] -> [B,5*T,D]: explicit time-major ordering.
        tokens = torch.stack(projected, dim=2).reshape(batch, history * NUM_MODALITIES, self.model_dim)
        valid_mask = torch.stack(checked_masks, dim=2).reshape(batch, history * NUM_MODALITIES)
        time_ids = torch.arange(history, device=external_rgb.device).repeat_interleave(NUM_MODALITIES)
        modality_ids = torch.arange(NUM_MODALITIES, device=external_rgb.device).repeat(history)
        tokens = self.final_norm(tokens + self.modality_embedding(modality_ids)
                                 + self.temporal_embedding(time_ids))
        return ConditionOutput(tokens, valid_mask, modality_ids, time_ids)
