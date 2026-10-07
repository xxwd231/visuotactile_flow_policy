"""Public token layout contract for the condition sequence."""

from dataclasses import dataclass
from enum import IntEnum

import torch


class Modality(IntEnum):
    EXTERNAL_RGB = 0
    WRIST_RGB = 1
    TACTILE_DEPTH_0 = 2
    TACTILE_DEPTH_1 = 3
    AGENT_POS = 4


MODALITY_ORDER = (
    "external_rgb",
    "wrist_rgb",
    "tactile_depth_0",
    "tactile_depth_1",
    "agent_pos",
)
NUM_MODALITIES = len(MODALITY_ORDER)
TOKEN_ORDER = "time_major"


@dataclass(frozen=True)
class ConditionOutput:
    """Tokens and metadata; time advances after every five modality tokens."""

    tokens: torch.Tensor  # [B, 5*T, model_dim]
    valid_mask: torch.Tensor  # [B, 5*T], bool; True means valid
    modality_ids: torch.Tensor  # [5*T], long
    time_ids: torch.Tensor  # [5*T], long; 0 is oldest
