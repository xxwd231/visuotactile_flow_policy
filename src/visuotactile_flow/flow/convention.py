"""Shared endpoint and direction contract for future objectives and solvers."""

from enum import StrEnum


class FlowConvention(StrEnum):
    NOISE_AT_ZERO = "noise_at_zero"
    NOISE_AT_ONE = "noise_at_one"

    @property
    def source_time(self) -> float:
        return float(self is FlowConvention.NOISE_AT_ONE)

    @property
    def target_time(self) -> float:
        return 1.0 - self.source_time

    @property
    def integration_sign(self) -> int:
        """Sign of time travel from source to target: +1 or -1."""
        return int(self.target_time - self.source_time)
