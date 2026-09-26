#!/usr/bin/env python3
"""Canonical full-head FedAvg with shared CodeBERT LoRA adapters."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

import run_fedopt_shared_head as base
import run_single_global_proto_triplet as common
from dual_global_proto_triplet import (
    aggregate_uniform_trainable_states,
    batch_order_sha256,
    require_max_only,
    trainable_state_sha256,
)
from federated_lora import (
    LoRAConfig,
    clone_named_state,
    inject_codebert_lora,
    lora_parameter_names,
    set_head_and_lora_trainable,
)
from run_all_g_full_function_matrix import FullFunctionALLG, model_forward, move_batch
from run_fedconv_private_head import build_loaders, prepare_clients
from run_fedproto_convpool_positive import TRAINABLE_PARAMETER_NAMES, load_trainable_state
from run_primevul_all_g_text import seed_everything, write_json


ALGORITHM = "canonical_fedavg_full_shared_head_with_federated_codebert_lora"
FIXED_LORA = LoRAConfig(rank=4, alpha=8.0, last_n_layers=2, targets=("query", "value"))


def parse_args() -> argparse.Namespace:
    front = argparse.ArgumentParser(add_help=False)
    front.add_argument("--initial_head_state", type=Path, required=True)
    known, remaining = front.parse_known_args()
    original = sys.argv
    try:
        sys.argv = [original[0], "--fedopt_arm", "fedavg", *remaining]
        args = base.parse_args()
    finally:
        sys.argv = original
    args.initial_head_state = known.initial_head_state
    return args


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    return clone_named_state(model, names)


def train_client_lora(
    model,
    loader,
    class_weights,
    bank,
    global_prototypes,
    client_control,
    server_control,
    use_scaffold: bool,
    args,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, object]:
    del bank, client_control, server_control
    if global_prototypes is not None or args.prototype_lambda != 0.0:
        raise ValueError("Federated LoRA screening is CE-only")
    if use_scaffold:
        raise ValueError("Federated LoRA screening does not use SCAFFOLD")
    model.train()
    # The frozen backbone remains deterministic. LoRA has no dropout in this
    # frozen configuration and remains trainable in eval mode.
    model.encoder.eval()
    named = dict(model.named_parameters())
    lora_names = set(lora_parameter_names(model))
    lora_params = [named[name] for name in sorted(lora_names)]
    head_params = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name not in lora_names
    ]
    if not lora_params or not head_params:
        raise RuntimeError("Expected both LoRA and canonical head parameters")
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
                logits, _, _ = model_forward(model, batch)
                classification = criterion(logits, batch["labels"])
            scaler.scale(classification / args.gradient_accumulation_steps).backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head_params + lora_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            size = int(batch["labels"].shape[0])
            totals["examples"] += size
            totals["classification_ce"] += float(classification.detach()) * size
    if optimizer_steps < 1:
        raise RuntimeError("No local AdamW update was performed")
    return {
        "loss": totals["classification_ce"] / max(1.0, totals["examples"]),
        "classification_ce": totals["classification_ce"] / max(1.0, totals["examples"]),
        "optimizer": "AdamW",
        "optimizer_steps": optimizer_steps,
        "head_weight_decay": args.weight_decay,
        "lora_weight_decay": 0.0,
        "lora_parameter_count": int(sum(parameter.numel() for parameter in lora_params)),
        "batch_order_sha256": batch_order_sha256(observed_ids, args.seed),
        "batches": len(loader) * args.local_epochs,
        "local_epoch_count": args.local_epochs,
    }


def load_head_initialization(
    path: Path,
    model: nn.Module,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("Unsupported canonical head initialization")
    if payload.get("seed") != seed or payload.get("token_pooling") != "max_only":
        raise ValueError("Canonical head initialization metadata mismatch")
    head_state = payload.get("model_state")
    if not isinstance(head_state, dict) or set(head_state) != set(TRAINABLE_PARAMETER_NAMES):
        raise ValueError("Canonical head initialization has unexpected scope")
    load_trainable_state(model, head_state)
    reloaded = clone_named_state(model, TRAINABLE_PARAMETER_NAMES)
    if trainable_state_sha256(reloaded) != payload.get("state_sha256"):
        raise ValueError("Canonical head initialization tensor hash mismatch")
    return reloaded, {
        "path": str(path),
        "head_state_sha256": payload["state_sha256"],
        "file_sha256": base.file_sha256(path),
    }


def compose_local_state(
    local_head_state: Mapping[str, torch.Tensor],
    initial_lora_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if set(local_head_state) != set(TRAINABLE_PARAMETER_NAMES):
        raise ValueError("Local reference has unexpected trainable scope")
    return {
        **{name: value.detach().cpu().clone() for name, value in local_head_state.items()},
        **{name: value.detach().cpu().clone() for name, value in initial_lora_state.items()},
    }


def main() -> None:
    args = parse_args()
    reference = common.validate_args(args)
    require_max_only(args.token_pooling)
    if args.fedopt_arm != "fedavg" or args.local_epochs < 1 or args.prototype_lambda != 0.0:
        raise ValueError("Federated LoRA experiment requires FedAvg, positive local_epochs, and CE-only training")
    if args.prepare_initial_only:
        raise ValueError("Use canonical FedAvg runner to prepare the shared head initialization")
    specs = sorted(args.client, key=lambda item: item[0])
    common.validate_group_metadata_dirs(specs)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "algorithm": ALGORITHM, "lora": vars(FIXED_LORA)}))
        return

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    model = FullFunctionALLG(
        args.encoder_path,
        "token_stitch",
        0,
        args.dropout,
        args.window_microbatch,
        args.token_pooling,
    )
    injected_modules = inject_codebert_lora(model, FIXED_LORA)
    model = model.to(device)
    trainable_names = set_head_and_lora_trainable(model, TRAINABLE_PARAMETER_NAMES)
    head_initial, head_manifest = load_head_initialization(
        args.initial_head_state, model, args.seed
    )
    initial_state = trainable_state(model)
    initial_lora_state = {
        name: initial_state[name].detach().cpu().clone()
        for name in lora_parameter_names(model)
    }
    # B=0 guarantees exact functional equality to the frozen CodeBERT base.
    if any(torch.count_nonzero(value).item() for name, value in initial_lora_state.items() if name.endswith(".lora_B")):
        raise RuntimeError("LoRA B must be zero at initialization")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_path, use_fast=True)
    collator = common.FullFunctionCollator(
        tokenizer, args.max_length, args.stride, args.max_windows
    )
    clients = prepare_clients(
        specs, tokenizer, collator, args.stride, args.max_windows, args.output_dir
    )
    loaders = build_loaders(
        clients, collator, args.batch_size, args.num_workers, args.seed, amp_enabled
    )
    groups = {client.name: common.validation_group_map(client) for client in clients}
    local_states = {
        client.name: compose_local_state(
            common.load_local_reference_state(args.local_reference_dir, client.name),
            initial_lora_state,
        )
        for client in clients
    }

    parameter_counts = {
        "lora": int(sum(dict(model.named_parameters())[name].numel() for name in lora_parameter_names(model))),
        "head": int(sum(dict(model.named_parameters())[name].numel() for name in TRAINABLE_PARAMETER_NAMES)),
        "communicated": int(sum(dict(model.named_parameters())[name].numel() for name in trainable_names)),
        "full_model": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    write_json(
        args.output_dir / "config.json",
        {
            "algorithm": ALGORITHM,
            "seed": args.seed,
            "token_pooling": "max_only",
            "backbone": "CodeBERT frozen except fixed shared LoRA",
            "lora": {
                "rank": FIXED_LORA.rank,
                "alpha": FIXED_LORA.alpha,
                "last_n_layers": FIXED_LORA.last_n_layers,
                "targets": list(FIXED_LORA.targets),
                "dropout": 0.0,
                "aggregation": "standard factor-wise equal-client FedAvg",
                "injected_modules": list(injected_modules),
            },
            "parameter_counts": parameter_counts,
            "trainable_parameter_names": list(trainable_names),
            "initial_head": head_manifest,
            "initial_trainable_state_sha256": trainable_state_sha256(initial_state),
            "loss": "inverse-frequency weighted CE only",
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "client_optimizer": "AdamW; head weight decay 1e-2; LoRA weight decay 0",
            "selection": "macro OOF validation MCC, macro NLL, earlier round",
            "test_isolation": "test opened after global round and thresholds are immutable",
            "reference_token_pooling": reference,
        },
    )

    # Reuse the canonical trajectory implementation with a dynamic trainable
    # state and the LoRA-aware optimizer. No other server/client logic changes.
    original_state_fn = base.scaffold.trainable_state
    original_train_fn = base.adamw.train_client_adamw
    base.scaffold.trainable_state = trainable_state
    base.adamw.train_client_adamw = train_client_lora
    try:
        trained = base.train_trajectory(
            model,
            clients,
            loaders,
            groups,
            initial_state,
            args.output_dir / "validation_predictions.jsonl",
            args,
            base.server_config(args),
            device,
            amp_enabled,
        )
    finally:
        base.scaffold.trainable_state = original_state_fn
        base.adamw.train_client_adamw = original_train_fn

    selected_round, round_summaries = base.fedcot_runner.select_global_round(
        trained["global_evaluations"], clients
    )
    selected_state = trained["global_states"][selected_round]
    selected_evaluations = trained["global_evaluations"][selected_round]
    validation_path = args.output_dir / "validation_predictions.jsonl"

    local_validation: dict[str, common.ValidationEvaluation] = {}
    for client in clients:
        load_trainable_state(model, local_states[client.name])
        local_validation[client.name] = common.evaluate_validation(
            model,
            loaders[client.name]["valid"],
            groups[client.name],
            client.name,
            "offline_local_reference",
            validation_path,
            args.seed,
            args.crossfit_folds,
            device,
            amp_enabled,
        )
    selection = {
        "selected_global_round": selected_round,
        "selected_state_sha256": trainable_state_sha256(selected_state),
        "round_summaries": round_summaries,
        "client_thresholds": {
            client.name: selected_evaluations[client.name].full_validation_threshold
            for client in clients
        },
        "test_accessed_during_training_or_selection": False,
    }
    write_json(args.output_dir / "selection_manifest.json", selection)
    torch.save(
        {"selected_global_round": selected_round, "model_state": selected_state},
        args.output_dir / "selected_global_model.pt",
    )

    guard = common.EvaluationGuard(selection_finalized=True)
    results: dict[str, dict[str, Any]] = {}
    for client in clients:
        load_trainable_state(model, local_states[client.name])
        local_test = base.scaffold.evaluate_test_all(
            model,
            loaders[client.name]["test"],
            guard,
            local_validation[client.name].full_validation_threshold,
            device,
            amp_enabled,
        )
        load_trainable_state(model, selected_state)
        shared_test = base.scaffold.evaluate_test_all(
            model,
            loaders[client.name]["test"],
            guard,
            selected_evaluations[client.name].full_validation_threshold,
            device,
            amp_enabled,
        )
        client_dir = args.output_dir / client.name
        client_dir.mkdir(parents=True, exist_ok=True)
        write_json(client_dir / "shared_test_predictions.json", shared_test["predictions"])
        results[client.name] = {
            "offline_local_reference": {
                "validation": common.validation_summary(local_validation[client.name]),
                "test": base.fedcot_runner.compact(local_test),
            },
            "global_shared": {
                "selected_round": selected_round,
                "validation": common.validation_summary(selected_evaluations[client.name]),
                "test": base.fedcot_runner.compact(shared_test),
            },
            "delta_mcc_vs_offline_local": float(shared_test["mcc"] - local_test["mcc"]),
        }
    metrics = ("accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc")
    names = [client.name for client in clients]
    deltas = [float(results[name]["delta_mcc_vs_offline_local"]) for name in names]
    output = {
        "algorithm": ALGORITHM,
        "selection": selection,
        "initial_head": head_manifest,
        "parameter_counts": parameter_counts,
        "clients": results,
        "aggregate": {
            source: {
                metric: float(np.mean([results[name][source]["test"][metric] for name in names]))
                for metric in metrics
            }
            for source in ("offline_local_reference", "global_shared")
        },
        "transfer": {
            "mean_delta_mcc": float(np.mean(deltas)),
            "min_delta_mcc": float(min(deltas)),
            "positive_transfer_count": int(sum(value > 0.0 for value in deltas)),
            "negative_transfer_count": int(sum(value < 0.0 for value in deltas)),
        },
        "test_evaluated_only_after_global_round_selection": True,
    }
    write_json(args.output_dir / "comparison.json", output)
    write_json(args.output_dir / "metrics.json", output)
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
