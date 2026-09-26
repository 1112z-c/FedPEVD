#!/usr/bin/env python3
"""Federated LoRA + Full Head FedAvg with client-local hard sample Triplet."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

import run_fedlora_shared_head as fedlora
import run_online_shared_sample_triplet_local_epochs_adamw as sample_triplet
from dual_global_proto_triplet import aggregate_uniform_trainable_states, batch_order_sha256, trainable_state_sha256
from run_all_g_full_function_matrix import model_forward, move_batch
from run_fedproto_convpool_positive import load_trainable_state
from run_primevul_all_g_text import seed_everything, write_json


ALGORITHM = "fedavg_federated_lora_full_shared_head_weighted_ce_hard_sample_triplet"


def parse_front_args() -> tuple[float, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--triplet_lambda", type=float, required=True)
    known, remaining = parser.parse_known_args()
    if known.triplet_lambda < 0.0:
        raise ValueError("triplet_lambda must be non-negative")
    return known.triplet_lambda, [sys.argv[0], *remaining]


def train_client_lora_triplet(
    model: nn.Module,
    loader,
    class_weights: torch.Tensor,
    bank,
    args,
    triplet_lambda: float,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, object]:
    model.train()
    model.encoder.eval()
    named = dict(model.named_parameters())
    lora_names = set(fedlora.lora_parameter_names(model))
    lora_params = [named[name] for name in sorted(lora_names)]
    head_params = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name not in lora_names
    ]
    if not lora_params or not head_params:
        raise RuntimeError("Expected both LoRA and Head parameters")
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "weight_decay": args.weight_decay},
            {"params": lora_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    totals: defaultdict[str, float] = defaultdict(float)
    observed_ids: list[str] = []
    optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)
    for _local_epoch in range(1, args.local_epochs + 1):
        for step, raw in enumerate(loader, 1):
            observed_ids.extend(str(value) for value in raw["sample_ids"])
            batch = move_batch(raw, device)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits, representations, _ = model_forward(model, batch)
                classification = criterion(logits, batch["labels"])
            triplet = classification.new_zeros(())
            audit = {
                "valid_triplets": 0.0,
                "active_rate": 0.0,
                "mean_positive_distance": 0.0,
                "mean_negative_distance": 0.0,
            }
            if bank and triplet_lambda > 0.0:
                with torch.cuda.amp.autocast(enabled=False):
                    triplet, audit = sample_triplet.sample_hard_triplet_loss(
                        representations.float(),
                        batch["labels"],
                        tuple(str(value) for value in raw["sample_ids"]),
                        bank,
                        args.triplet_margin,
                    )
            objective = classification + triplet_lambda * triplet
            scaler.scale(objective / args.gradient_accumulation_steps).backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head_params + lora_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            size = int(batch["labels"].shape[0])
            totals["examples"] += size
            totals["loss"] += float(objective.detach()) * size
            totals["classification_ce"] += float(classification.detach()) * size
            totals["triplet"] += float(triplet.detach()) * size
            for key, value in audit.items():
                totals[key] += float(value) * size
    if optimizer_steps < 1:
        raise RuntimeError("No local AdamW update was performed")
    divisor = max(1.0, totals["examples"])
    return {
        **{key: value / divisor for key, value in totals.items() if key != "examples"},
        "optimizer": "AdamW",
        "optimizer_steps": optimizer_steps,
        "head_weight_decay": args.weight_decay,
        "lora_weight_decay": 0.0,
        "lora_parameter_count": int(sum(parameter.numel() for parameter in lora_params)),
        "batch_order_sha256": batch_order_sha256(observed_ids, args.seed),
        "batches": len(loader) * args.local_epochs,
        "local_epoch_count": args.local_epochs,
    }


def train_trajectory(
    model,
    clients,
    loaders,
    groups,
    initial_state: Mapping[str, torch.Tensor],
    validation_path: Path,
    args,
    config,
    device: torch.device,
    amp_enabled: bool,
    triplet_lambda: float,
) -> dict[str, Any]:
    base = fedlora.base
    global_state = base.clone_state(initial_state)
    optimizer_state = base.initialize_server_state(global_state, config)
    history: list[dict[str, Any]] = []
    global_states: dict[int, dict[str, torch.Tensor]] = {}
    global_evaluations = {}
    banks: dict[str, object] = {}
    checkpoint_dir = args.output_dir / "global_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    for round_index in range(1, args.rounds + 1):
        local_states = {}
        next_banks = {}
        train = {}
        for index, client in enumerate(clients):
            seed_everything(args.seed + round_index * 10_000 + index)
            load_trainable_state(model, global_state)
            train[client.name] = train_client_lora_triplet(
                model,
                loaders[client.name]["train"],
                client.class_weights,
                banks.get(client.name, {}) if round_index > args.warmup_rounds else {},
                args,
                triplet_lambda,
                device,
                amp_enabled,
            )
            local_states[client.name] = fedlora.trainable_state(model)
            next_banks[client.name] = sample_triplet.collect_train_bank(
                model, loaders[client.name]["train"], device, amp_enabled
            )
        proposal = aggregate_uniform_trainable_states(local_states)
        accepted, optimizer_state, diagnostics = base.apply_server_update(
            global_state, proposal, optimizer_state, config
        )
        evaluations = base.evaluate_global_validation(
            model,
            accepted,
            clients,
            loaders,
            groups,
            validation_path,
            round_index,
            args,
            device,
            amp_enabled,
        )
        previous_sha = trainable_state_sha256(global_state)
        proposal_sha = trainable_state_sha256(proposal)
        accepted_sha = trainable_state_sha256(accepted)
        global_state = base.clone_state(accepted)
        banks = next_banks
        global_states[round_index] = base.clone_state(global_state)
        global_evaluations[round_index] = evaluations
        torch.save(
            {
                "round": round_index,
                "fedopt_arm": args.fedopt_arm,
                "server_optimizer": config.name,
                "model_state": base.clone_state(global_state),
                "server_optimizer_state": {
                    name: base.clone_state(values) for name, values in optimizer_state.items()
                },
                "state_sha256": accepted_sha,
            },
            checkpoint_dir / f"global_round_{round_index:02d}.pt",
        )
        row = {
            "round": round_index,
            "fedopt_arm": args.fedopt_arm,
            "server_optimizer": config.name,
            "previous_state_sha256": previous_sha,
            "fedavg_proposal_sha256": proposal_sha,
            "accepted_state_sha256": accepted_sha,
            "server_diagnostics": diagnostics,
            "sample_triplet_active": round_index > args.warmup_rounds,
            "validation": {
                client.name: fedlora.common.validation_summary(evaluations[client.name])
                for client in clients
            },
            "train": train,
        }
        history.append(row)
        write_json(args.output_dir / "history.json", history)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return {
        "history": history,
        "global_states": global_states,
        "global_evaluations": global_evaluations,
    }


def main() -> None:
    triplet_lambda, rewritten_argv = parse_front_args()
    original_argv = sys.argv
    original_algorithm = fedlora.ALGORITHM
    original_trajectory = fedlora.base.train_trajectory
    fedlora.ALGORITHM = ALGORITHM

    def patched_trajectory(*args, **kwargs):
        return train_trajectory(*args, **kwargs, triplet_lambda=triplet_lambda)

    fedlora.base.train_trajectory = patched_trajectory
    sys.argv = rewritten_argv
    try:
        fedlora.main()
    finally:
        sys.argv = original_argv
        fedlora.ALGORITHM = original_algorithm
        fedlora.base.train_trajectory = original_trajectory

    output_index = original_argv.index("--output_dir") + 1
    config_path = Path(original_argv[output_index]) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "algorithm": ALGORITHM,
            "loss": "inverse-frequency weighted CE + client-local hard sample Triplet",
            "triplet_lambda": triplet_lambda,
            "triplet_margin": 1.0,
            "sample_bank": "client-local train-only; cap 256/class; never uploaded",
            "triplet_mining": "farthest same-class positive; nearest foreign-class negative",
        }
    )
    write_json(config_path, config)


if __name__ == "__main__":
    main()
