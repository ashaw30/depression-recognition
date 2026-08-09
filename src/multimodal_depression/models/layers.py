"""Core layers: temporal alignment, encoding, attention, and fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMBlock(nn.Module):
    """Bidirectional LSTM followed by LayerNorm."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim // 2, bidirectional=True, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        return self.layer_norm(x)


class AttentionPooling(nn.Module):
    """Learnable temporal attention pooling."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)
        return (x * weights).sum(dim=1)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with residual connection."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        scale = self.head_dim ** 0.5
        attn = torch.softmax((q @ k.transpose(-2, -1)) / scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch, seq_len, self.dim)
        return self.proj(out) + x


class CausalTemporalAttention(nn.Module):
    """Causal temporal attention producing a sequence-level vector."""

    def __init__(self, dim: int, max_seq_len: int = 1200) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        attn = (q @ k.transpose(-2, -1)) / (dim ** 0.5)
        attn = attn.masked_fill(self.mask[:seq_len, :seq_len] == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        return (attn @ v).mean(dim=1)


class CrossModalAttention(nn.Module):
    """Bidirectional cross-modal attention aligned with the manuscript.

    Audio and AU sequences interact via MultiheadAttention (audio<-AU, AU<-audio).
    Original and attended features are pooled and concatenated:

        f_cross = [f_a; f_a_tilde; f_v; f_v_tilde]  in R^{4H}
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        nhead = max(1, min(num_heads, hidden_dim // 8))
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim * 4
        self.audio_cross = nn.MultiheadAttention(
            hidden_dim, nhead, batch_first=True, dropout=0.1
        )
        self.au_cross = nn.MultiheadAttention(
            hidden_dim, nhead, batch_first=True, dropout=0.1
        )
        self.norm_audio = nn.LayerNorm(hidden_dim)
        self.norm_au = nn.LayerNorm(hidden_dim)
        self.audio_pool = AttentionPooling(hidden_dim)
        self.au_pool = AttentionPooling(hidden_dim)

    def forward(
        self, audio_seq: torch.Tensor, au_seq: torch.Tensor
    ) -> torch.Tensor:
        audio_ctx, _ = self.audio_cross(audio_seq, au_seq, au_seq)
        au_ctx, _ = self.au_cross(au_seq, audio_seq, audio_seq)

        f_a = self.audio_pool(self.norm_audio(audio_seq))
        f_v = self.au_pool(self.norm_au(au_seq))
        f_a_tilde = self.audio_pool(self.norm_audio(audio_ctx))
        f_v_tilde = self.au_pool(self.norm_au(au_ctx))

        # Paper: [f_a; f_a_tilde; f_v; f_v_tilde]
        return torch.cat([f_a, f_a_tilde, f_v, f_v_tilde], dim=-1)


class VectorCrossModal(nn.Module):
    """Vector-level cross-modal fusion (used by some ablation variants)."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn_weights = nn.Parameter(torch.randn(2, 2))
        self.proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, audio: torch.Tensor, au: torch.Tensor) -> torch.Tensor:
        features = torch.stack([audio, au], dim=1)
        sim = torch.matmul(features, features.transpose(1, 2))
        attn = torch.softmax(sim * self.attn_weights, dim=-1)
        fused = torch.matmul(attn, features)
        return self.proj(fused.reshape(fused.size(0), -1))


class DynamicFusionLayer(nn.Module):
    """Gender-conditioned gating (demographic-conditioned fusion)."""

    def __init__(self, hidden_dim: int, gender_emb_dim: int = 32) -> None:
        super().__init__()
        self.gender_emb = nn.Embedding(2, gender_emb_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + gender_emb_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
            nn.Sigmoid(),
        )
        self.res_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, gender: torch.Tensor) -> torch.Tensor:
        gender_idx = gender.long().clamp(0, 1).squeeze(-1)
        gender_feat = self.gender_emb(gender_idx)
        if x.dim() == 3:
            gender_feat = gender_feat.unsqueeze(1)
        gate = self.gate(torch.cat([x, gender_feat], dim=-1))
        out = x + gate * self.res_proj(x)
        return out.squeeze(1) if out.dim() == 3 else out.squeeze()


class TimeAlign(nn.Module):
    """Align variable-length sequences to a fixed length (crop or interpolate)."""

    def __init__(self, target_length: int) -> None:
        super().__init__()
        self.target_length = target_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len == self.target_length:
            return x
        if seq_len > self.target_length:
            return x[:, : self.target_length, :]
        return F.interpolate(
            x.permute(0, 2, 1),
            size=self.target_length,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)
