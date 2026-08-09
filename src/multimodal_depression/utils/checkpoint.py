"""Utilities for keeping the best model parameters during training."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_best_model(
    model: nn.Module,
    score: float,
    best_score: float | None,
    path: str | Path,
    *,
    mode: str = "max",
    extra: dict[str, Any] | None = None,
    min_delta: float = 0.0,
) -> tuple[bool, float, dict[str, torch.Tensor] | None]:
    """Save ``model`` when ``score`` improves over ``best_score``.

    Parameters
    ----------
    model:
        Model whose ``state_dict`` should be stored.
    score:
        Current validation metric (e.g. AUC, F1).
    best_score:
        Best metric so far; pass ``None`` on the first call.
    path:
        Destination ``.pth`` / ``.pt`` file.
    mode:
        ``"max"`` for metrics to maximize (AUC/F1), ``"min"`` for loss.
    extra:
        Optional metadata stored alongside weights (epoch, metrics, ...).
    min_delta:
        Minimum improvement required to overwrite the checkpoint.

    Returns
    -------
    improved, new_best_score, state_dict_or_none
    """
    path = Path(path)
    if mode not in {"max", "min"}:
        raise ValueError("mode must be 'max' or 'min'")

    if best_score is None:
        improved = True
    elif mode == "max":
        improved = score > best_score + min_delta
    else:
        improved = score < best_score - min_delta

    if not improved:
        return False, float(best_score), None

    state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    payload: dict[str, Any] = {
        "model_state_dict": state_dict,
        "score": float(score),
        "mode": mode,
    }
    if extra:
        payload["extra"] = extra

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return True, float(score), state_dict


def load_model_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint saved by :func:`save_best_model` into ``model``."""
    path = Path(path)
    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        meta = payload
    else:
        state_dict = payload
        meta = {"model_state_dict": state_dict}

    model.load_state_dict(state_dict, strict=strict)
    return meta


class BestModelSaver:
    """Track the best score and persist the corresponding model weights."""

    def __init__(
        self,
        path: str | Path = "best_model.pth",
        *,
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        self.path = Path(path)
        self.mode = mode
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.best_state: dict[str, torch.Tensor] | None = None

    def update(
        self,
        model: nn.Module,
        score: float,
        *,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        improved, self.best_score, state = save_best_model(
            model,
            score,
            self.best_score,
            self.path,
            mode=self.mode,
            extra=extra,
            min_delta=self.min_delta,
        )
        if improved and state is not None:
            self.best_state = state
        return improved

    def load_best(self, model: nn.Module) -> nn.Module:
        if self.best_state is not None:
            model.load_state_dict(deepcopy(self.best_state))
            return model
        if self.path.exists():
            load_model_checkpoint(model, self.path)
            return model
        raise FileNotFoundError(f"No best checkpoint available at {self.path}")
