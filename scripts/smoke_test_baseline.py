#!/usr/bin/env python
"""Instantiate Baseline, run a forward pass, and demo best-model saving."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal_depression.config import ModelConfig
from multimodal_depression.models import MultiModalDepressionClassifier
from multimodal_depression.utils import BestModelSaver


def main() -> None:
    config = ModelConfig()
    model = MultiModalDepressionClassifier(config)
    model.eval()

    batch = 2
    audio = torch.randn(batch, config.audio_seq_len, 17)
    au = torch.randn(batch, config.au_seq_len, 17)
    gender = torch.tensor([[0.0], [1.0]])

    with torch.no_grad():
        probs = model(audio, au, gender)

    print("Baseline forward OK")
    print(f"  audio: {tuple(audio.shape)}")
    print(f"  au:    {tuple(au.shape)}")
    print(f"  cross-modal out dim: {model.fused_dim} (= 4 * hidden_dim)")
    print(f"  out:   {tuple(probs.shape)} -> {probs.squeeze().tolist()}")
    print(
        "Pipeline: TimeAlign -> BiLSTM -> MultiHeadAttention "
        "-> CrossModalAttention -> DynamicFusionLayer(gender) -> classifier"
    )

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "best_model.pth"
        saver = BestModelSaver(ckpt, mode="max")
        assert saver.update(model, score=0.80, extra={"epoch": 1})
        assert not saver.update(model, score=0.79, extra={"epoch": 2})
        assert saver.update(model, score=0.91, extra={"epoch": 3})
        saver.load_best(model)
        print(f"BestModelSaver OK -> {ckpt.name}, best_score={saver.best_score}")


if __name__ == "__main__":
    main()
