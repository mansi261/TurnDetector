"""Train and select the English Whisper Tiny baseline."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import sqlite3
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from turn_detector.data_loader import CanonicalPipecatDataset
from turn_detector.models import (
    WaveformFeatureExtractor,
    WhisperTurnDetector,
    load_model_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/english_whisper_tiny_v1.json"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))
    parser.add_argument("--output-name")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256() -> str:
    digest = hashlib.sha256()
    roots = (Path("src"), Path("configs"))
    paths = [
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    paths.extend(path for path in Path(".").glob("requirements*.txt") if path.is_file())
    for path in sorted(paths):
        digest.update(path.as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def command_output(*command: str) -> str | None:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def split_identity(index_path: Path, language: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    with sqlite3.connect(index_path) as connection:
        for split in ("train", "validation", "evaluation"):
            digest = hashlib.sha256()
            counts = {"hold": 0, "end": 0}
            rows = connection.execute(
                """
                SELECT decision_id, label FROM decisions
                WHERE split = ? AND language = ? ORDER BY decision_id
                """,
                (split, language),
            )
            count = 0
            for decision_id, label in rows:
                digest.update(decision_id.encode())
                digest.update(b"\n")
                counts[label] += 1
                count += 1
            result[split] = {
                "records": count,
                "labels": counts,
                "decision_id_sha256": digest.hexdigest(),
            }
        train_ids = {
            row[0]
            for row in connection.execute(
                "SELECT source_id FROM decisions WHERE split = 'train' AND language = ?",
                (language,),
            )
        }
        overlap = sum(
            row[0] in train_ids
            for row in connection.execute(
                "SELECT source_id FROM decisions WHERE split = 'validation' AND language = ?",
                (language,),
            )
        )
        metadata = dict(connection.execute("SELECT key, value FROM index_metadata"))
    result["train_validation_id_overlap"] = overlap
    result["index_metadata"] = metadata
    return result


def environment_record(device: torch.device) -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "numpy", "pyarrow", "soundfile"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device": str(device),
        "torch_mps_available": torch.backends.mps.is_available(),
        "torch_cuda_available": torch.cuda.is_available(),
        "packages": packages,
    }


def write_run_manifest(
    output: Path,
    config_path: Path,
    config: dict[str, Any],
    device: torch.device,
) -> None:
    index_path = Path(config["dataset"]["index_path"])
    snapshot_manifest = Path(config["dataset"]["snapshot_root"]) / "snapshot_manifest.json"
    dirty_diff = command_output("git", "diff", "--no-ext-diff", "HEAD") or ""
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "invocation": [sys.executable, *sys.argv],
        "config": config,
        "config_sha256": file_sha256(config_path),
        "canonical_index_sha256": file_sha256(index_path),
        "snapshot_manifest_sha256": file_sha256(snapshot_manifest),
        "source_sha256": source_sha256(),
        "splits": split_identity(index_path, config["dataset"]["language"]),
        "git_head": command_output("git", "rev-parse", "HEAD"),
        "git_dirty_diff_sha256": hashlib.sha256(dirty_diff.encode()).hexdigest(),
        "environment": environment_record(device),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_resume_state(
    output: Path,
    checkpoint: Path,
    config_path: Path,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[WhisperTurnDetector, list[dict[str, Any]], int, str]:
    checkpoint = checkpoint.resolve()
    if checkpoint.parent.parent != output:
        raise ValueError("resume checkpoint must belong to the selected run directory")
    if (output / "results.json").exists():
        raise ValueError("cannot resume a completed run")
    manifest_path = output / "run_manifest.json"
    log_path = output / "training_log.jsonl"
    if not manifest_path.exists() or not log_path.exists() or not checkpoint.exists():
        raise FileNotFoundError("resume requires the manifest, training log, and checkpoint")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config_sha256"] != file_sha256(config_path):
        raise ValueError("current configuration differs from the original run")
    index_path = Path(config["dataset"]["index_path"])
    if manifest["canonical_index_sha256"] != file_sha256(index_path):
        raise ValueError("canonical index differs from the original run")

    model, payload = load_model_checkpoint(checkpoint, device=device)
    if payload["model_config"] != config["model"]:
        raise ValueError("checkpoint model configuration differs from the run configuration")
    completed_epoch = int(payload["training_state"]["epoch"])
    history = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [record["epoch"] for record in history] != list(range(1, completed_epoch + 1)):
        raise ValueError("training log does not end at the checkpoint epoch")
    if completed_epoch >= int(config["training"]["epochs"]):
        raise ValueError("checkpoint already reached the configured epoch count")
    checkpoint_hash = file_sha256(checkpoint)
    event = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "invocation": [sys.executable, *sys.argv],
        "resume_checkpoint": checkpoint.as_posix(),
        "resume_checkpoint_sha256": checkpoint_hash,
        "completed_epoch": completed_epoch,
        "next_epoch": completed_epoch + 1,
        "config_sha256": file_sha256(config_path),
        "source_sha256": source_sha256(),
        "git_head": command_output("git", "rev-parse", "HEAD"),
        "environment": environment_record(device),
        "rng_state_restored": False,
        "resume_boundary": "A new optimizer and scheduler begin at the frozen-to-unfrozen epoch boundary.",
    }
    with (output / "resume_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    return model, history, completed_epoch, checkpoint_hash


class FeatureCollator:
    def __init__(
        self,
        backbone: str,
        window_seconds: float,
        sample_rate: int,
        revision: str | None = None,
        local_files_only: bool = False,
    ):
        self.extractor = WaveformFeatureExtractor(
            backbone, window_seconds, sample_rate, revision, local_files_only
        )

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        features = self.extractor([example["waveform"] for example in examples])
        return {
            "input_features": features,
            "labels": torch.stack([example["label"] for example in examples]),
            "ids": [example["id"] for example in examples],
        }


def make_loader(
    config: dict[str, Any],
    *,
    split: str,
    collator: FeatureCollator,
    smoke: bool,
) -> tuple[CanonicalPipecatDataset, DataLoader]:
    data = config["dataset"]
    training = config["training"]
    max_key = "smoke_train_examples" if split == "train" else "smoke_validation_examples"
    dataset = CanonicalPipecatDataset(
        data["index_path"],
        split=split,
        language=data["language"],
        window_seconds=data["window_seconds"],
        sample_rate=data["sample_rate"],
        shuffle=split == "train",
        shuffle_buffer=training["shuffle_buffer"],
        seed=training["seed"],
        max_examples=training[max_key] if smoke else None,
        normalize_waveform=bool(data.get("normalize_waveform", False)),
    )
    batch_size = (
        training["batch_size"] if split == "train" else training["validation_batch_size"]
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0 if smoke else training["num_workers"],
        collate_fn=collator,
        drop_last=False,
        pin_memory=False,
    )
    return dataset, loader


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = probabilities >= threshold
    positives = labels == 1
    negatives = ~positives
    true_positive = int(np.sum(predictions & positives))
    false_positive = int(np.sum(predictions & negatives))
    true_negative = int(np.sum(~predictions & negatives))
    false_negative = int(np.sum(~predictions & positives))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    false_positive_rate = false_positive / max(1, false_positive + true_negative)
    accuracy = (true_positive + true_negative) / max(1, len(labels))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    positive_scores = probabilities[positives]
    negative_scores = probabilities[negatives]
    if len(positive_scores) and len(negative_scores):
        combined = np.concatenate((positive_scores, negative_scores))
        order = np.argsort(combined, kind="stable")
        ranks = np.empty(len(combined), dtype=np.float64)
        position = 0
        while position < len(combined):
            end = position + 1
            while end < len(combined) and combined[order[end]] == combined[order[position]]:
                end += 1
            ranks[order[position:end]] = (position + 1 + end) / 2
            position = end
        rank_sum = np.sum(ranks[: len(positive_scores)])
        auc = (rank_sum - len(positive_scores) * (len(positive_scores) + 1) / 2) / (
            len(positive_scores) * len(negative_scores)
        )
    else:
        auc = float("nan")
    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "end_recall": recall,
        "hold_false_positive_rate": false_positive_rate,
        "f1": f1,
        "auroc": float(auc),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    maximum_false_positive_rate: float,
) -> tuple[float, dict[str, float]]:
    candidates = np.unique(np.concatenate(([0.0, 1.0], probabilities)))
    feasible = []
    for threshold in candidates:
        metrics = binary_metrics(labels, probabilities, float(threshold))
        if metrics["hold_false_positive_rate"] <= maximum_false_positive_rate:
            feasible.append((metrics["end_recall"], metrics["f1"], -float(threshold), metrics))
    if not feasible:
        raise RuntimeError("no threshold satisfies the false-positive constraint")
    best = max(feasible, key=lambda item: item[:3])
    return best[3]["threshold"], best[3]


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    losses = 0.0
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input_features"].to(device)
            targets = batch["labels"].to(device)
            logits = model(inputs)
            losses += float(criterion(logits, targets).cpu())
            labels.append(targets.cpu().numpy())
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    all_labels = np.concatenate(labels)
    all_probabilities = np.concatenate(probabilities)
    metrics = binary_metrics(all_labels, all_probabilities, 0.5)
    metrics["loss"] = losses / len(all_labels)
    metrics["records"] = int(len(all_labels))
    return metrics, all_labels, all_probabilities


def configure_optimizer(model: WhisperTurnDetector, config: dict[str, Any], frozen: bool):
    training = config["training"]
    model.freeze_encoder(frozen)
    if frozen:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": training["head_learning_rate"]}]
    else:
        groups = [
            {"params": model.encoder.parameters(), "lr": training["encoder_learning_rate"]},
            {
                "params": list(model.attention.parameters()) + list(model.classifier.parameters()),
                "lr": training["finetune_head_learning_rate"],
            },
        ]
    return torch.optim.AdamW(groups, weight_decay=training["weight_decay"])


def scheduler_for(optimizer, steps: int, warmup_fraction: float):
    warmup_steps = max(1, round(steps * warmup_fraction))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def save_checkpoint(
    path: Path,
    model: WhisperTurnDetector,
    *,
    epoch: int,
    metrics: dict[str, Any],
    threshold: float = 0.5,
) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(
        model.checkpoint_payload(
            threshold=threshold,
            training_state={"epoch": epoch, "metrics": metrics},
        ),
        temporary,
    )
    temporary.replace(path)


def train(
    config_path: Path,
    output_root: Path,
    smoke: bool,
    requested_device: str,
    output_name: str | None = None,
    resume_from: Path | None = None,
) -> Path:
    config_path = config_path.resolve()
    config = read_config(config_path)
    training = config["training"]
    seed_everything(training["seed"])
    device = select_device(requested_device)
    run_name = output_name or ("smoke" if smoke else "full")
    output = (output_root / config["experiment_name"] / run_name).resolve()
    if resume_from is None:
        if output.exists() and any(path.is_file() for path in output.rglob("*")):
            raise FileExistsError(f"run directory is not empty: {output}")
        (output / "checkpoints").mkdir(parents=True, exist_ok=True)
        write_run_manifest(output, config_path, config, device)
    elif not output.exists():
        raise FileNotFoundError(f"run directory does not exist: {output}")

    collator = FeatureCollator(
        config["model"]["backbone"],
        config["dataset"]["window_seconds"],
        config["dataset"]["sample_rate"],
        config["model"].get("backbone_revision"),
    )
    train_dataset, train_loader = make_loader(
        config, split="train", collator=collator, smoke=smoke
    )
    _, validation_loader = make_loader(
        config, split="validation", collator=collator, smoke=smoke
    )
    epochs = training["smoke_epochs"] if smoke else training["epochs"]
    resume_checkpoint_hash = None
    if resume_from is None:
        model = WhisperTurnDetector.from_backbone(config["model"]).to(device)
        history = []
        start_epoch = 0
        prior_elapsed_seconds = 0.0
    else:
        model, history, start_epoch, resume_checkpoint_hash = load_resume_state(
            output, resume_from, config_path, config, device
        )
        if start_epoch >= epochs:
            raise ValueError("resume checkpoint already reached this run's epoch count")
        prior_elapsed_seconds = float(history[-1]["elapsed_seconds"])
    best_loss = min(
        (float(record["validation"]["loss"]) for record in history),
        default=float("inf"),
    )
    log_path = output / "training_log.jsonl"
    optimizer = None
    scheduler = None
    frozen_state = None
    started = time.perf_counter()

    for epoch in range(start_epoch, epochs):
        train_dataset.set_epoch(epoch)
        frozen = epoch < training["freeze_encoder_epochs"]
        if frozen != frozen_state:
            optimizer = configure_optimizer(model, config, frozen)
            remaining_epochs = training["freeze_encoder_epochs"] if frozen else epochs - epoch
            steps = max(1, math.ceil(len(train_dataset) / training["batch_size"]) * remaining_epochs)
            scheduler = scheduler_for(optimizer, steps, training["warmup_fraction"])
            frozen_state = frozen
        assert optimizer is not None and scheduler is not None
        model.train()
        epoch_loss = 0.0
        records = 0
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            inputs = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with context:
                logits = model(inputs)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"])
            optimizer.step()
            scheduler.step()
            batch_records = len(labels)
            epoch_loss += float(loss.detach().cpu()) * batch_records
            records += batch_records
            if step % 100 == 0:
                print(f"epoch={epoch + 1} step={step} loss={epoch_loss / records:.5f}")

        if records != len(train_dataset):
            raise RuntimeError(
                f"epoch {epoch + 1} processed {records} of {len(train_dataset)} training records"
            )
        validation, _, _ = evaluate_model(model, validation_loader, device)
        epoch_record = {
            "epoch": epoch + 1,
            "encoder_frozen": frozen,
            "train_loss": epoch_loss / max(1, records),
            "train_records": records,
            "validation": validation,
            "elapsed_seconds": prior_elapsed_seconds + time.perf_counter() - started,
        }
        history.append(epoch_record)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(epoch_record, sort_keys=True) + "\n")
        save_checkpoint(
            output / "checkpoints/latest.pt", model, epoch=epoch + 1, metrics=validation
        )
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            save_checkpoint(
                output / "checkpoints/best.pt", model, epoch=epoch + 1, metrics=validation
            )
        print(json.dumps(epoch_record, indent=2))

    best_model, best_payload = load_model_checkpoint(
        output / "checkpoints/best.pt", device=device
    )
    validation, labels, probabilities = evaluate_model(best_model, validation_loader, device)
    threshold, selected_metrics = select_threshold(
        labels,
        probabilities,
        config["threshold"]["maximum_hold_false_positive_rate"],
    )
    validation["selected_threshold"] = threshold
    validation["selected_threshold_metrics"] = selected_metrics
    save_checkpoint(
        output / "checkpoints/best.pt",
        best_model,
        epoch=best_payload["training_state"]["epoch"],
        metrics=validation,
        threshold=threshold,
    )
    np.savez_compressed(output / "validation_predictions.npz", labels=labels, probabilities=probabilities)
    results = {
        "run": run_name,
        "device": str(device),
        "history": history,
        "best_epoch": best_payload["training_state"]["epoch"],
        "validation": validation,
        "best_checkpoint": "checkpoints/best.pt",
        "best_checkpoint_sha256": file_sha256(output / "checkpoints/best.pt"),
        "elapsed_seconds": prior_elapsed_seconds + time.perf_counter() - started,
        "resumed_from": resume_from.resolve().as_posix() if resume_from else None,
        "resume_checkpoint_sha256": resume_checkpoint_hash,
    }
    (output / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    reloaded, _ = load_model_checkpoint(output / "checkpoints/best.pt", device=device)
    first_batch = next(iter(validation_loader))
    with torch.inference_mode():
        before = best_model(first_batch["input_features"].to(device)).cpu()
        after = reloaded(first_batch["input_features"].to(device)).cpu()
    if not torch.allclose(before, after, atol=1e-6, rtol=1e-5):
        raise RuntimeError("checkpoint reload parity failed")
    print(f"run complete: {output}")
    return output


def main() -> None:
    args = parse_args()
    train(
        args.config,
        args.output_root,
        args.smoke,
        args.device,
        args.output_name,
        args.resume_from,
    )


if __name__ == "__main__":
    main()
