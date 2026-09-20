from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import time

import torch
import torch.nn as nn


def _build_sigmoid_mlp(input_dim: int, hidden_dim: int, mlp_layers: int) -> nn.Sequential:
    if mlp_layers < 1:
        raise ValueError(f"Unsupported mlp_layers={mlp_layers}; expected an integer >= 1.")
    layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
    for _ in range(mlp_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
    layers.extend([nn.Linear(hidden_dim, 1), nn.Sigmoid()])
    return nn.Sequential(*layers)


class TokenAgnosticGateMLP(nn.Module):
    """Lambda gate over sorted top-k logits from SLM/LM."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, mlp_layers: int = 1) -> None:
        super().__init__()
        self.net = _build_sigmoid_mlp(input_dim=input_dim, hidden_dim=hidden_dim, mlp_layers=mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class KLStepSample:
    features: torch.Tensor
    teacher_logits: torch.Tensor
    slm_logits: torch.Tensor
    lm_logits: torch.Tensor
    slm_base_logits: Optional[torch.Tensor] = None


@dataclass
class SampleBuildStats:
    examples_seen: int = 0
    examples_used: int = 0
    steps_used: int = 0
    skipped_empty_teacher: int = 0
    skipped_vocab_mismatch: int = 0


def build_gate_features(slm_logits: torch.Tensor, lm_logits: torch.Tensor, gate_top_k: int) -> torch.Tensor:
    """Token-agnostic gate features = sorted top-k logits from each model."""
    gate_top_k = max(1, min(gate_top_k, int(slm_logits.shape[-1]), int(lm_logits.shape[-1])))
    slm_top = torch.topk(slm_logits, k=gate_top_k, dim=-1).values
    lm_top = torch.topk(lm_logits, k=gate_top_k, dim=-1).values
    return torch.cat([slm_top, lm_top], dim=-1).to(dtype=torch.float32)


def build_contrastive_gate_features(
    slm_ctx_logits: torch.Tensor,
    slm_base_logits: torch.Tensor,
    lm_logits: torch.Tensor,
    gate_top_k: int,
) -> torch.Tensor:
    delta_logits = slm_ctx_logits - slm_base_logits
    return build_gate_features(delta_logits, lm_logits, gate_top_k=gate_top_k)


def contrastive_prob_fused_log_probs(
    slm_logits: torch.Tensor,
    slm_base_logits: torch.Tensor,
    lm_logits: torch.Tensor,
    lambda_w: torch.Tensor,
) -> torch.Tensor:
    lm_probs = torch.softmax(lm_logits, dim=-1)
    slm_probs = torch.softmax(slm_logits, dim=-1)
    slm_base_probs = torch.softmax(slm_base_logits, dim=-1)
    fused_probs = lm_probs + lambda_w * (slm_probs - slm_base_probs)
    fused_probs = torch.clamp(fused_probs, min=1e-12)
    fused_probs = fused_probs / fused_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return torch.log(fused_probs)


def build_attn_ratio_and_logit_topk_features(
    slm_logits: torch.Tensor,
    lm_logits: torch.Tensor,
    *,
    gate_top_k: int,
    ctx_attn_ratio: torch.Tensor,
) -> torch.Tensor:
    """Token-agnostic top-k logits plus one scalar context-attention ratio."""
    logit_feats = build_gate_features(slm_logits, lm_logits, gate_top_k=gate_top_k)
    ratio_feat = ctx_attn_ratio.reshape(1).to(dtype=torch.float32, device=logit_feats.device)
    return torch.cat([logit_feats, ratio_feat], dim=-1)


def gate_feature_names(gate_feature_type: str, gate_top_k: int) -> List[str]:
    gate_top_k = max(1, int(gate_top_k))
    if gate_feature_type == "logit_topk":
        return (
            [f"slm_top_logit_{i+1}" for i in range(gate_top_k)]
            + [f"lm_top_logit_{i+1}" for i in range(gate_top_k)]
        )
    if gate_feature_type == "attn_ratio_and_logit_topk":
        return (
            [f"slm_top_logit_{i+1}" for i in range(gate_top_k)]
            + [f"lm_top_logit_{i+1}" for i in range(gate_top_k)]
            + ["ctx_attn_ratio"]
        )
    return []


def kl_teacher_to_fused(
    teacher_logits: torch.Tensor,
    slm_logits: torch.Tensor,
    lm_logits: torch.Tensor,
    lambda_w: torch.Tensor,
    *,
    fusion_mode: str = "weighted_sum",
    slm_base_logits: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    teacher_probs = torch.softmax(teacher_logits, dim=-1)
    if fusion_mode in {"contrastive", "contrastive_prob"}:
        if slm_base_logits is None:
            raise ValueError(f"slm_base_logits is required when fusion_mode={fusion_mode!r}.")
        if fusion_mode == "contrastive_prob":
            fused_log_probs = contrastive_prob_fused_log_probs(slm_logits, slm_base_logits, lm_logits, lambda_w)
        else:
            fused_logits = lm_logits + lambda_w * (slm_logits - slm_base_logits)
            fused_log_probs = torch.log_softmax(fused_logits, dim=-1)
    else:
        fused_logits = lambda_w * slm_logits + (1.0 - lambda_w) * lm_logits
        fused_log_probs = torch.log_softmax(fused_logits, dim=-1)
    teacher_log_probs = torch.log(teacher_probs.clamp(min=1e-12))
    return torch.sum(teacher_probs * (teacher_log_probs - fused_log_probs), dim=-1)


def train_gate_with_kl(
    gate: TokenAgnosticGateMLP,
    train_samples: Sequence[KLStepSample],
    *,
    val_samples: Sequence[KLStepSample] | None = None,
    epochs: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 64,
    device: str = "cpu",
    fusion_mode: str = "weighted_sum",
) -> List[Dict[str, float]]:
    if not train_samples:
        raise ValueError("No train samples provided.")

    gate = gate.to(device)
    gate.train()
    opt = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=weight_decay)

    history: List[Dict[str, float]] = []
    n = len(train_samples)

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        perm = torch.randperm(n).tolist()
        total_loss = 0.0
        updates_this_epoch = 0

        opt.zero_grad(set_to_none=True)
        steps_since_update = 0

        for idx, sample_idx in enumerate(perm, start=1):
            sample = train_samples[sample_idx]
            x = sample.features.to(device=device).unsqueeze(0)
            teacher_logits = sample.teacher_logits.to(device=device)
            slm_logits = sample.slm_logits.to(device=device)
            lm_logits = sample.lm_logits.to(device=device)
            slm_base_logits = sample.slm_base_logits.to(device=device) if sample.slm_base_logits is not None else None

            lambda_w = gate(x).view(1)
            loss = kl_teacher_to_fused(
                teacher_logits,
                slm_logits,
                lm_logits,
                lambda_w,
                fusion_mode=fusion_mode,
                slm_base_logits=slm_base_logits,
            )
            loss = loss / batch_size
            loss.backward()
            total_loss += float(loss.item()) * batch_size

            steps_since_update += 1
            if steps_since_update >= batch_size or idx == n:
                nn.utils.clip_grad_norm_(gate.parameters(), max_norm=1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                steps_since_update = 0
                updates_this_epoch += 1

        train_loss = total_loss / n
        row: Dict[str, float] = {
            "epoch": float(epoch),
            "train_kl": float(train_loss),
            "updates": float(updates_this_epoch),
            "epoch_time_s": float(time.perf_counter() - epoch_start),
        }

        if val_samples:
            gate.eval()
            with torch.no_grad():
                val_total = 0.0
                for sample in val_samples:
                    x = sample.features.to(device=device).unsqueeze(0)
                    teacher_logits = sample.teacher_logits.to(device=device)
                    slm_logits = sample.slm_logits.to(device=device)
                    lm_logits = sample.lm_logits.to(device=device)
                    slm_base_logits = sample.slm_base_logits.to(device=device) if sample.slm_base_logits is not None else None
                    lambda_w = gate(x).view(1)
                    val_total += float(
                        kl_teacher_to_fused(
                            teacher_logits,
                            slm_logits,
                            lm_logits,
                            lambda_w,
                            fusion_mode=fusion_mode,
                            slm_base_logits=slm_base_logits,
                        ).item()
                    )
            row["val_kl"] = val_total / len(val_samples)
            gate.train()

        history.append(row)

    return history
