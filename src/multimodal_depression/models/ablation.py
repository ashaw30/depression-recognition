"""Ablation variants corresponding to the paper component analysis."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from multimodal_depression.config import ModelConfig

from .classifier import MultiModalDepressionClassifier
from .layers import DynamicFusionLayer


def build_baseline(config: ModelConfig) -> MultiModalDepressionClassifier:
    return MultiModalDepressionClassifier(config)


def build_legacy_mean(config: ModelConfig) -> MultiModalDepressionClassifier:
    """Remove multi-head self-attention; use BiLSTM + mean pooling only."""

    class LegacyMeanModel(MultiModalDepressionClassifier):
        def __init__(self, cfg: ModelConfig) -> None:
            super().__init__(cfg)
            self.fusion = DynamicFusionLayer(cfg.hidden_dim, gender_emb_dim=32)
            self.classifier = nn.Sequential(
                nn.Linear(cfg.hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(64, 1),
            )

        def forward(self, audio, au, gender, reading_scalars=None):
            gender = gender.view(-1, 1)
            audio = self.audio_align(audio)
            au = self.au_align(au)
            audio_feat = self.audio_lstm(audio).mean(dim=1)
            au_feat = self.au_lstm(au).mean(dim=1)
            cross_feat = self.vector_cross(audio_feat, au_feat)
            fused = self.fusion(cross_feat.unsqueeze(1), gender)
            return torch.sigmoid(self.classifier(fused))

    return LegacyMeanModel(config)


def build_no_time_align(config: ModelConfig) -> MultiModalDepressionClassifier:
    """Remove temporal alignment (TimeAlign)."""

    class NoTimeAlignModel(MultiModalDepressionClassifier):
        def encode_audio_sequence(self, audio: torch.Tensor) -> torch.Tensor:
            hidden = self.audio_lstm(audio)
            hidden = self.audio_attn(hidden)
            return self.dropout(hidden)

        def encode_au_sequence(self, au: torch.Tensor) -> torch.Tensor:
            hidden = self.au_lstm(au)
            hidden = self.au_attn(hidden)
            return self.dropout(hidden)

    return NoTimeAlignModel(config)


def build_no_cross_modal(config: ModelConfig) -> MultiModalDepressionClassifier:
    """Replace cross-modal attention with concatenation."""

    class NoCrossModalModel(MultiModalDepressionClassifier):
        def __init__(self, cfg: ModelConfig) -> None:
            super().__init__(cfg)
            self.fusion = DynamicFusionLayer(cfg.hidden_dim * 2, gender_emb_dim=32)
            self.classifier = nn.Sequential(
                nn.Linear(cfg.hidden_dim * 2, 64),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(64, 1),
            )

        def forward(self, audio, au, gender, reading_scalars=None):
            gender = gender.view(-1, 1)
            audio_feat = self.encode_audio(audio)
            au_feat = self.encode_au(au)
            fused = torch.cat([audio_feat, au_feat], dim=1)
            fused = self.fusion(fused.unsqueeze(1), gender)
            return torch.sigmoid(self.classifier(fused))

    return NoCrossModalModel(config)


def build_concat_fusion(config: ModelConfig) -> MultiModalDepressionClassifier:
    """Replace gender-conditioned gating with simple gender concatenation."""

    class ConcatFusionModel(MultiModalDepressionClassifier):
        def __init__(self, cfg: ModelConfig) -> None:
            super().__init__(cfg)
            self.fusion = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Linear(self.fused_dim + 1, 64),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(64, 1),
            )

        def forward(self, audio, au, gender, reading_scalars=None):
            gender = gender.view(-1, 1)
            audio_seq = self.encode_audio_sequence(audio)
            au_seq = self.encode_au_sequence(au)
            cross_feat = self.cross_modal_attn(audio_seq, au_seq)
            cross_feat = self.apply_reading_scalars(cross_feat, reading_scalars)
            fused = torch.cat([cross_feat, gender], dim=1)
            return torch.sigmoid(self.classifier(fused))

    return ConcatFusionModel(config)


def build_no_gender(config: ModelConfig) -> MultiModalDepressionClassifier:
    """Disable gender conditioning (zero gender)."""

    class NoGenderModel(MultiModalDepressionClassifier):
        def forward(self, audio, au, gender, reading_scalars=None):
            gender = gender.view(-1, 1)
            audio_seq = self.encode_audio_sequence(audio)
            au_seq = self.encode_au_sequence(au)
            cross_feat = self.cross_modal_attn(audio_seq, au_seq)
            cross_feat = self.apply_reading_scalars(cross_feat, reading_scalars)
            fused = self.fusion(
                cross_feat.unsqueeze(1), torch.zeros_like(gender)
            )
            return torch.sigmoid(self.classifier(fused))

    return NoGenderModel(config)


ABLATION_BUILDERS: dict[str, Callable[[ModelConfig], nn.Module]] = {
    "Baseline": build_baseline,
    "LegacyMean": build_legacy_mean,
    "NoTimeAlign": build_no_time_align,
    "NoCrossModal": build_no_cross_modal,
    "ConcatFusion": build_concat_fusion,
    "NoGender": build_no_gender,
}


def build_ablation_model(name: str, config: ModelConfig) -> nn.Module:
    if name not in ABLATION_BUILDERS:
        raise KeyError(f"Unknown ablation: {name}. Options: {list(ABLATION_BUILDERS)}")
    return ABLATION_BUILDERS[name](config)
