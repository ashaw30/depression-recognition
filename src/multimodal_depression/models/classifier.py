"""Baseline multimodal depression classifier."""

from __future__ import annotations

import torch
import torch.nn as nn

from multimodal_depression.config import ModelConfig
from multimodal_depression.constants import AUDIO_DIM, AU_DIM

from .layers import (
    AttentionPooling,
    BiLSTMBlock,
    CausalTemporalAttention,
    CrossModalAttention,
    DynamicFusionLayer,
    MultiHeadAttention,
    TimeAlign,
    VectorCrossModal,
)


class MultiModalDepressionClassifier(nn.Module):
    """Audio (MFCC + prosody) + facial AU + gender multimodal classifier.

    Baseline forward path:
    TimeAlign -> BiLSTM -> Multi-Head Self-Attention
    -> CrossModalAttention ([f_a; f_a~; f_v; f_v~] in R^{4H})
    -> DynamicFusionLayer (gender) -> MLP classifier
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.fused_dim = config.hidden_dim * 4

        self.audio_align = TimeAlign(config.audio_seq_len)
        self.au_align = TimeAlign(config.au_seq_len)

        self.audio_lstm = BiLSTMBlock(AUDIO_DIM, config.hidden_dim)
        self.au_lstm = BiLSTMBlock(AU_DIM, config.hidden_dim)

        self.audio_attn = MultiHeadAttention(config.hidden_dim, config.num_heads)
        self.au_attn = MultiHeadAttention(config.hidden_dim, config.num_heads)
        self.audio_time_attn = CausalTemporalAttention(
            config.hidden_dim, max_seq_len=config.audio_seq_len
        )
        self.au_time_attn = CausalTemporalAttention(
            config.hidden_dim, max_seq_len=config.au_seq_len
        )

        self.audio_pool = AttentionPooling(config.hidden_dim)
        self.au_pool = AttentionPooling(config.hidden_dim)

        # Kept for ablation variants that still use vector-level fusion.
        self.vector_cross = VectorCrossModal(config.hidden_dim)
        self.cross_modal_attn = CrossModalAttention(
            config.hidden_dim, num_heads=max(1, config.num_heads // 2)
        )
        self.fusion = DynamicFusionLayer(self.fused_dim, gender_emb_dim=32)
        self.dropout = nn.Dropout(config.dropout)

        if config.use_reading_scalars:
            self.scalar_proj = nn.Sequential(
                nn.Linear(config.n_reading_scalars, config.hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 4, self.fused_dim),
            )
        else:
            self.scalar_proj = None

        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
        )

    def encode_audio_sequence(self, audio: torch.Tensor) -> torch.Tensor:
        audio = self.audio_align(audio)
        hidden = self.audio_lstm(audio)
        hidden = self.audio_attn(hidden)
        return self.dropout(hidden)

    def encode_au_sequence(self, au: torch.Tensor) -> torch.Tensor:
        au = self.au_align(au)
        hidden = self.au_lstm(au)
        hidden = self.au_attn(hidden)
        return self.dropout(hidden)

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        return self.encode_audio_sequence(audio).mean(dim=1)

    def encode_au(self, au: torch.Tensor) -> torch.Tensor:
        return self.encode_au_sequence(au).mean(dim=1)

    def apply_reading_scalars(
        self, cross_feat: torch.Tensor, reading_scalars: torch.Tensor | None
    ) -> torch.Tensor:
        if self.scalar_proj is None or reading_scalars is None:
            return cross_feat
        return cross_feat + self.scalar_proj(reading_scalars)

    def forward(
        self,
        audio: torch.Tensor,
        au: torch.Tensor,
        gender: torch.Tensor,
        reading_scalars: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gender = gender.view(-1, 1)
        audio_seq = self.encode_audio_sequence(audio)
        au_seq = self.encode_au_sequence(au)
        cross_feat = self.cross_modal_attn(audio_seq, au_seq)
        cross_feat = self.apply_reading_scalars(cross_feat, reading_scalars)
        fused = self.fusion(cross_feat.unsqueeze(1), gender)
        return torch.sigmoid(self.classifier(fused))
