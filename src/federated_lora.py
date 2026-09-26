#!/usr/bin/env python3
"""Minimal LoRA utilities for the canonical CodeBERT federated experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 4
    alpha: float = 8.0
    last_n_layers: int = 2
    targets: tuple[str, ...] = ("query", "value")

    def validate(self) -> None:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0.0:
            raise ValueError("LoRA alpha must be positive")
        if self.last_n_layers <= 0:
            raise ValueError("LoRA last_n_layers must be positive")
        if not self.targets or any(name not in {"query", "value"} for name in self.targets):
            raise ValueError("Only CodeBERT attention query/value targets are supported")


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank update.

    The update is zero at initialization because ``lora_B`` is initialized to
    zero. This makes the initial function exactly equal to the frozen backbone.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear requires nn.Linear")
        if rank <= 0 or alpha <= 0.0:
            raise ValueError("Invalid LoRA rank/alpha")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base(inputs)
        update = functional.linear(functional.linear(inputs, self.lora_A), self.lora_B)
        return base + update * self.scaling


def inject_codebert_lora(model: nn.Module, config: LoRAConfig) -> tuple[str, ...]:
    """Inject LoRA into Q/V projections of the final CodeBERT layers."""

    config.validate()
    layers = model.encoder.encoder.layer
    if config.last_n_layers > len(layers):
        raise ValueError("LoRA layer count exceeds CodeBERT depth")
    injected: list[str] = []
    start = len(layers) - config.last_n_layers
    for layer_index in range(start, len(layers)):
        attention = layers[layer_index].attention.self
        for target in config.targets:
            original = getattr(attention, target)
            if isinstance(original, LoRALinear):
                raise RuntimeError(f"LoRA already injected at layer {layer_index} {target}")
            if not isinstance(original, nn.Linear):
                raise TypeError(f"Unexpected CodeBERT projection type: {type(original)!r}")
            setattr(attention, target, LoRALinear(original, config.rank, config.alpha))
            injected.append(f"encoder.encoder.layer.{layer_index}.attention.self.{target}")
    return tuple(injected)


def lora_parameter_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name, _ in model.named_parameters()
            if name.endswith(".lora_A") or name.endswith(".lora_B")
        )
    )


def set_head_and_lora_trainable(
    model: nn.Module,
    head_parameter_names: Iterable[str],
) -> tuple[str, ...]:
    """Freeze the backbone and enable exactly the canonical head plus LoRA."""

    head = set(head_parameter_names)
    lora = set(lora_parameter_names(model))
    allowed = head | lora
    if not lora:
        raise RuntimeError("No LoRA parameters were injected")
    actual_names = {name for name, _ in model.named_parameters()}
    missing = allowed - actual_names
    if missing:
        raise KeyError(f"Model is missing trainable parameters: {sorted(missing)}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in allowed)
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if actual != allowed:
        raise RuntimeError(f"Unexpected trainable scope: {sorted(actual ^ allowed)}")
    return tuple(sorted(allowed))


def clone_named_state(model: nn.Module, names: Iterable[str]) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    selected = tuple(names)
    missing = set(selected) - set(state)
    if missing:
        raise KeyError(f"State is missing keys: {sorted(missing)}")
    return {name: state[name].detach().cpu().clone() for name in selected}

