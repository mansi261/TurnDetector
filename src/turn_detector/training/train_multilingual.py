"""Train the isolated 14-language Whisper Tiny turn detector."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from turn_detector.data_loader import MultilingualPipecatDataset
from turn_detector.evaluation.metrics import calibration, score_distribution
from turn_detector.models import WaveformFeatureExtractor, load_model_checkpoint
from turn_detector.training.train import (
    binary_metrics,
    environment_record,
    read_config,
    scheduler_for,
    seed_everything,
    select_device,
    source_sha256,
)
from turn_detector.utils import command_output, file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/multilingual_whisper_tiny_v1.json")
    )
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))
    parser.add_argument("--output-name")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--initialization-checkpoint", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


class MultilingualFeatureCollator:
    def __init__(self, config: dict[str, Any]):
        dataset = config["dataset"]
        model = config["model"]
        self.extractor = WaveformFeatureExtractor(
            model["backbone"], dataset["window_seconds"], dataset["sample_rate"],
            model.get("backbone_revision"),
        )

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "input_features": self.extractor([example["waveform"] for example in examples]),
            "labels": torch.stack([example["label"] for example in examples]),
            "ids": [example["id"] for example in examples],
            "languages": [example["language"] for example in examples],
            "synthetic": [example["synthetic"] for example in examples],
        }


def split_identity(index_path: Path, languages: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    placeholders = ",".join("?" for _ in languages)
    with sqlite3.connect(index_path) as connection:
        for split in ("train", "validation", "evaluation", "holdout"):
            digest = hashlib.sha256()
            counts: dict[str, dict[str, int]] = {}
            for decision_id, language, label in connection.execute(
                f"""
                SELECT decision_id, language, label FROM decisions
                WHERE split = ? AND language IN ({placeholders}) ORDER BY decision_id
                """,
                (split, *languages),
            ):
                digest.update(decision_id.encode())
                digest.update(b"\n")
                values = counts.setdefault(language, {"hold": 0, "end": 0})
                values[label] += 1
            result[split] = {
                "records": sum(sum(labels.values()) for labels in counts.values()),
                "by_language": counts,
                "decision_id_sha256": digest.hexdigest(),
            }
        result["index_metadata"] = dict(
            connection.execute("SELECT key, value FROM index_metadata")
        )
    return result


def smoke_ids(index_path: Path, languages: list[str], split: str) -> list[str]:
    result = []
    with sqlite3.connect(index_path) as connection:
        for language in languages:
            for label in ("hold", "end"):
                row = connection.execute(
                    """
                    SELECT decision_id FROM decisions
                    WHERE source = 'pipecat' AND split = ? AND language = ? AND label = ?
                    ORDER BY decision_id LIMIT 1
                    """,
                    (split, language, label),
                ).fetchone()
                if row is None:
                    raise RuntimeError(f"missing smoke example for {split}/{language}/{label}")
                result.append(row[0])
    return result


def make_dataset(
    config: dict[str, Any], split: str, *, smoke: bool, start_example: int = 0
) -> MultilingualPipecatDataset:
    dataset = config["dataset"]
    languages = dataset["languages"]
    return MultilingualPipecatDataset(
        dataset["index_path"],
        split=split,
        languages=languages,
        window_seconds=dataset["window_seconds"],
        sample_rate=dataset["sample_rate"],
        shuffle=split == "train",
        seed=config["training"]["seed"],
        decision_ids=(smoke_ids(Path(dataset["index_path"]), languages, split) if smoke else None),
        start_example=start_example,
        normalize_waveform=bool(dataset.get("normalize_waveform", False)),
    )


def make_loader(
    dataset: MultilingualPipecatDataset,
    config: dict[str, Any],
    collator: MultilingualFeatureCollator,
    split: str,
) -> DataLoader:
    training = config["training"]
    batch_size = (
        training["batch_size"] if split == "train" else training["validation_batch_size"]
    )
    generator = torch.Generator()
    generator.manual_seed(training["seed"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=training["num_workers"],
        collate_fn=collator,
        drop_last=False,
        pin_memory=False,
        generator=generator,
    )


def language_metrics(
    labels: np.ndarray, probabilities: np.ndarray, languages: np.ndarray
) -> dict[str, Any]:
    results = {}
    losses = []
    for language in sorted(set(languages.tolist())):
        mask = languages == language
        language_labels = labels[mask]
        language_probabilities = probabilities[mask]
        values = binary_metrics(language_labels, language_probabilities, 0.5)
        clipped = np.clip(language_probabilities, 1e-7, 1 - 1e-7)
        loss = -np.mean(
            language_labels * np.log(clipped)
            + (1 - language_labels) * np.log(1 - clipped)
        )
        values["loss"] = float(loss)
        values["records"] = int(mask.sum())
        values["calibration"] = calibration(language_labels, language_probabilities)
        values["score_distribution"] = score_distribution(
            language_labels, language_probabilities
        )
        results[language] = values
        losses.append(float(loss))
    return {
        "by_language": results,
        "macro_language_loss": float(np.mean(losses)),
        "macro_language_accuracy": float(np.mean([v["accuracy"] for v in results.values()])),
        "macro_language_precision": float(np.mean([v["precision"] for v in results.values()])),
        "macro_language_end_recall": float(
            np.mean([v["end_recall"] for v in results.values()])
        ),
    }


def evaluate(model, loader: DataLoader, device: torch.device) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    all_labels = []
    all_probabilities = []
    all_languages = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["input_features"].to(device))
            all_labels.append(batch["labels"].numpy())
            all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
            all_languages.extend(batch["languages"])
    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probabilities)
    languages = np.asarray(all_languages)
    micro = binary_metrics(labels, probabilities, 0.5)
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    micro["loss"] = float(
        -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
    )
    micro["records"] = len(labels)
    micro["calibration"] = calibration(labels, probabilities)
    micro["score_distribution"] = score_distribution(labels, probabilities)
    result = {"micro": micro, **language_metrics(labels, probabilities, languages)}
    return result, {"labels": labels, "probabilities": probabilities, "languages": languages}


def capture_rng_state() -> dict[str, Any]:
    result = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        result["mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if (
        "mps" in state
        and torch.backends.mps.is_available()
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(state["mps"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    *,
    epoch_index: int,
    examples_completed: int,
    global_step: int,
    loss_sum: float,
    metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = model.checkpoint_payload(
        threshold=0.5,
        training_state={
            "epoch": epoch_index,
            "examples_completed": examples_completed,
            "global_step": global_step,
            "loss_sum": loss_sum,
            "metrics": metrics,
        },
    )
    payload["resume_state"] = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": capture_rng_state(),
    }
    return payload


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_manifest(
    output: Path,
    config_path: Path,
    config: dict[str, Any],
    device: torch.device,
    initialization_checkpoint: Path,
) -> None:
    index = Path(config["dataset"]["index_path"])
    diff = command_output("git", "diff", "--no-ext-diff", "HEAD") or ""
    artifacts = {
        "config": config_path,
        "canonical_index": index,
        "audit_report": Path(config["dataset"]["audit_report"]),
        "split_manifest": Path(config["dataset"]["split_manifest"]),
        "initialization_checkpoint": initialization_checkpoint,
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "invocation": [sys.executable, *sys.argv],
        "config": config,
        "artifact_sha256": {name: file_sha256(path) for name, path in artifacts.items()},
        "source_sha256": source_sha256(),
        "git_head": command_output("git", "rev-parse", "HEAD"),
        "git_dirty_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "splits": split_identity(index, config["dataset"]["languages"]),
        "environment": environment_record(device),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_initial_model(
    config: dict[str, Any], device: torch.device, checkpoint: Path
):
    path = checkpoint
    actual_hash = file_sha256(path)
    if actual_hash != config["initialization"]["checkpoint_sha256"]:
        raise RuntimeError("initialization checkpoint hash differs from the frozen contract")
    model, payload = load_model_checkpoint(path, device=device)
    if payload["model_config"] != config["model"]:
        raise RuntimeError("initialization model architecture differs from multilingual config")
    return model


def optimizer_and_scheduler(model, config: dict[str, Any], total_steps: int):
    model.freeze_encoder(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config["training"]["head_learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = scheduler_for(
        optimizer, total_steps, config["training"]["warmup_fraction"]
    )
    return optimizer, scheduler


def verify_resume_payload(
    path: Path, config: dict[str, Any], device: torch.device, total_steps: int
) -> None:
    model, payload = load_model_checkpoint(path, device=device)
    optimizer, scheduler = optimizer_and_scheduler(model, config, total_steps)
    optimizer.load_state_dict(payload["resume_state"]["optimizer"])
    scheduler.load_state_dict(payload["resume_state"]["scheduler"])
    restore_rng_state(payload["resume_state"]["rng"])
    if scheduler.last_epoch != payload["resume_state"]["scheduler"]["last_epoch"]:
        raise RuntimeError("scheduler resume verification failed")


def train(
    config_path: Path,
    output_root: Path,
    *,
    smoke: bool,
    requested_device: str,
    output_name: str | None,
    resume_from: Path | None,
    initialization_checkpoint: Path | None,
) -> Path:
    config_path = config_path.resolve()
    config = read_config(config_path)
    training = config["training"]
    if training["num_workers"] != 0:
        raise ValueError("exact multilingual resume currently requires num_workers=0")
    if not training["encoder_frozen"]:
        raise ValueError("the v1 multilingual contract requires a frozen encoder")
    seed_everything(training["seed"])
    device = select_device(requested_device)
    run_name = output_name or ("smoke" if smoke else "full")
    output = (output_root / config["experiment_name"] / run_name).resolve()
    initialization_checkpoint = (
        initialization_checkpoint
        or Path(config["initialization"]["checkpoint"])
    ).resolve()
    checkpoints = output / "checkpoints"
    if resume_from is None:
        if output.exists() and any(path.is_file() for path in output.rglob("*")):
            raise FileExistsError(f"run directory is not empty: {output}")
        checkpoints.mkdir(parents=True, exist_ok=True)
        write_manifest(output, config_path, config, device, initialization_checkpoint)
    else:
        if not output.exists() or (output / "results.json").exists():
            raise ValueError("resume requires an existing incomplete run directory")

    collator = MultilingualFeatureCollator(config)
    full_train = make_dataset(config, "train", smoke=smoke)
    validation = make_dataset(config, "validation", smoke=smoke)
    epochs = 1
    steps_per_epoch = math.ceil(full_train.total_examples / training["batch_size"])
    total_steps = steps_per_epoch * epochs
    started = time.perf_counter()
    if resume_from is None:
        model = load_initial_model(config, device, initialization_checkpoint)
        optimizer, scheduler = optimizer_and_scheduler(model, config, total_steps)
        epoch_index = 0
        examples_completed = 0
        global_step = 0
        previous_elapsed = 0.0
        epoch_loss_sum = 0.0
    else:
        resume_from = resume_from.resolve()
        if resume_from.parent.parent != output:
            raise ValueError("resume checkpoint must belong to the selected run")
        model, payload = load_model_checkpoint(resume_from, device=device)
        optimizer, scheduler = optimizer_and_scheduler(model, config, total_steps)
        optimizer.load_state_dict(payload["resume_state"]["optimizer"])
        scheduler.load_state_dict(payload["resume_state"]["scheduler"])
        restore_rng_state(payload["resume_state"]["rng"])
        state = payload["training_state"]
        epoch_index = int(state["epoch"])
        examples_completed = int(state["examples_completed"])
        global_step = int(state["global_step"])
        previous_elapsed = float(state.get("elapsed_seconds", 0.0))
        epoch_loss_sum = float(state.get("loss_sum", 0.0))
        with (output / "resume_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                        "checkpoint": resume_from.as_posix(),
                        "checkpoint_sha256": file_sha256(resume_from),
                        "epoch_index": epoch_index,
                        "examples_completed": examples_completed,
                        "global_step": global_step,
                        "optimizer_scheduler_rng_restored": True,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    if epoch_index >= epochs:
        raise ValueError("checkpoint already completed all epochs")
    full_train.set_epoch(epoch_index, start_example=examples_completed)
    train_loader = make_loader(full_train, config, collator, "train")
    validation_loader = make_loader(validation, config, collator, "validation")
    model.train()
    epoch_records = examples_completed
    interval = (
        training["smoke_checkpoint_interval_steps"]
        if smoke
        else training["checkpoint_interval_steps"]
    )
    for batch in train_loader:
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
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            model.parameters(), training["gradient_clip_norm"]
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("non-finite gradient norm")
        optimizer.step()
        scheduler.step()
        batch_records = len(labels)
        epoch_loss_sum += float(loss.detach().cpu()) * batch_records
        epoch_records += batch_records
        global_step += 1
        if global_step % 100 == 0 or smoke:
            print(
                f"epoch=1 step={global_step}/{total_steps} "
                f"records={epoch_records}/{full_train.total_examples} "
                f"loss={epoch_loss_sum / max(1, epoch_records):.5f}"
            )
        if global_step % interval == 0:
            state = checkpoint_payload(
                model, optimizer, scheduler, epoch_index=epoch_index,
                examples_completed=epoch_records, global_step=global_step,
                loss_sum=epoch_loss_sum, metrics=None,
            )
            state["training_state"]["elapsed_seconds"] = (
                previous_elapsed + time.perf_counter() - started
            )
            save_checkpoint(checkpoints / "latest.pt", state)

    if epoch_records != full_train.total_examples:
        raise RuntimeError(
            f"processed {epoch_records} of {full_train.total_examples} training records"
        )
    validation_result, predictions = evaluate(model, validation_loader, device)
    elapsed = previous_elapsed + time.perf_counter() - started
    epoch_record = {
        "epoch": 1,
        "encoder_frozen": True,
        "train_loss": epoch_loss_sum / max(1, epoch_records),
        "train_records": epoch_records,
        "global_step": global_step,
        "validation": validation_result,
        "elapsed_seconds": elapsed,
    }
    (output / "training_log.jsonl").write_text(
        json.dumps(epoch_record, sort_keys=True) + "\n", encoding="utf-8"
    )
    final_payload = checkpoint_payload(
        model, optimizer, scheduler, epoch_index=1, examples_completed=epoch_records,
        global_step=global_step, loss_sum=epoch_loss_sum, metrics=validation_result,
    )
    final_payload["training_state"]["elapsed_seconds"] = elapsed
    save_checkpoint(checkpoints / "latest.pt", final_payload)
    save_checkpoint(checkpoints / "best.pt", final_payload)
    np.savez_compressed(output / "validation_predictions.npz", **predictions)

    reloaded, _ = load_model_checkpoint(checkpoints / "best.pt", device=device)
    first_batch = next(iter(validation_loader))
    with torch.inference_mode():
        before = model(first_batch["input_features"].to(device)).cpu()
        after = reloaded(first_batch["input_features"].to(device)).cpu()
    if not torch.allclose(before, after, atol=1e-6, rtol=1e-5):
        raise RuntimeError("checkpoint reload parity failed")
    verify_resume_payload(checkpoints / "latest.pt", config, device, total_steps)

    results = {
        "run": run_name,
        "device": str(device),
        "single_seed": training["seed"],
        "history": [epoch_record],
        "best_epoch": 1,
        "selection_metric": "macro_language_validation_loss",
        "validation": validation_result,
        "best_checkpoint": "checkpoints/best.pt",
        "best_checkpoint_sha256": file_sha256(checkpoints / "best.pt"),
        "latest_checkpoint_sha256": file_sha256(checkpoints / "latest.pt"),
        "elapsed_seconds": elapsed,
        "resumed_from": resume_from.as_posix() if resume_from else None,
        "reload_probability_parity": True,
        "resume_state_verified": True,
    }
    (output / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    return output


def main() -> None:
    args = parse_args()
    train(
        args.config, args.output_root, smoke=args.smoke,
        requested_device=args.device, output_name=args.output_name,
        resume_from=args.resume_from,
        initialization_checkpoint=args.initialization_checkpoint,
    )


if __name__ == "__main__":
    main()
