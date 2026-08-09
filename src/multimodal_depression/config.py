"""Model hyperparameters for the Baseline architecture."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    num_heads: int = 8
    stamp_size: int = 125
    dropout: float = 0.3
    use_reading_scalars: bool = False
    n_reading_scalars: int = 10

    @property
    def au_seq_len(self) -> int:
        return self.stamp_size

    @property
    def audio_seq_len(self) -> int:
        return 4 * self.stamp_size
