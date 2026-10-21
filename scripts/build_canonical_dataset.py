#!/usr/bin/env python3
"""Convert Pipecat training and LiveKit evaluation metadata to one schema.

The output is decision-level: one Pipecat clip becomes one training decision,
while every LiveKit silence span becomes one evaluation decision. LiveKit audio
is exposed only through 200 ms after silence begins to avoid terminal-silence
length leaking the label.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


SCHEMA_VERSION = 1
LABEL_IDS = {"hold": 0, "end": 1}


@dataclass(frozen=True)
class CanonicalDecision:
    schema_version: int
    decision_id: str
    source: str
    source_id: str
    group_id: str
    split: str
    language: str
    label: str
    label_id: int
    audio_path: str
    source_duration_s: float
    pause_index: int | None
    silence_start_s: float | None
    silence_end_s: float | None
    decision_time_s: float
    window_start_s: float
    window_end_s: float
    left_padding_s: float
    target_sample_rate_hz: int
    target_channels: int
    model_transcript: str | None
    model_dialogue_context: str | None
    synthetic: bool | None
    provenance: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipecat-root",
        type=Path,
        default=Path("data/pipecat_smart_turn_v3_2_sample"),
    )
    parser.add_argument(
        "--livekit-root",
        type=Path,
        default=Path("data/LiveKit_50_English_Evaluation_Examples"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/canonical"))
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--score-point", type=float, default=0.2)
    return parser.parse_args()


def load_pipecat_durations(root: Path) -> dict[str, float]:
    validation = root / "audio_validation.tsv"
    durations: dict[str, float] = {}
    with validation.open(encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            path = Path(row["relative_audio_path"])
            durations[path.name] = float(row["duration_seconds"])
    return durations


def pipecat_decisions(root: Path, window_seconds: float) -> Iterator[CanonicalDecision]:
    durations = load_pipecat_durations(root)
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            audio_path = root / row["relative_audio_path"]
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            duration = durations[audio_path.name]
            label = "end" if row["endpoint_bool"].lower() == "true" else "hold"
            yield CanonicalDecision(
                schema_version=SCHEMA_VERSION,
                decision_id=f"pipecat:{row['id']}",
                source="pipecat",
                source_id=row["id"],
                group_id=row["id"],
                split="train",
                language="en" if row["language"] == "eng" else row["language"],
                label=label,
                label_id=LABEL_IDS[label],
                audio_path=audio_path.resolve().as_posix(),
                source_duration_s=duration,
                pause_index=None,
                silence_start_s=None,
                silence_end_s=None,
                decision_time_s=duration,
                window_start_s=max(0.0, duration - window_seconds),
                window_end_s=duration,
                left_padding_s=max(0.0, window_seconds - duration),
                target_sample_rate_hz=16000,
                target_channels=1,
                model_transcript=None,
                model_dialogue_context=None,
                synthetic=row["synthetic"].lower() == "true",
                provenance=row["source_dataset"],
            )


def livekit_group_id(source_id: str) -> str:
    marker = "__user_turn_"
    return source_id.split(marker, 1)[0] if marker in source_id else source_id


def livekit_decisions(
    root: Path, window_seconds: float, score_point: float
) -> Iterator[CanonicalDecision]:
    payload = json.loads((root / "examples.json").read_text(encoding="utf-8"))
    for example in payload["examples"]:
        audio_path = root / example["audio_path"]
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        pauses = example.get("pause_decisions") or [
            {
                **span,
                "label": "end" if index == len(example["silence_spans"]) - 1 else "hold",
            }
            for index, span in enumerate(example["silence_spans"])
        ]
        for pause_index, pause in enumerate(pauses):
            label = "end" if pause["label"] in {"end", "eot"} else "hold"
            decision_time = min(float(pause["end"]), float(pause["start"]) + score_point)
            yield CanonicalDecision(
                schema_version=SCHEMA_VERSION,
                decision_id=f"livekit:{example['id']}:pause_{pause_index:02d}",
                source="livekit",
                source_id=example["id"],
                group_id=livekit_group_id(example["id"]),
                split="evaluation",
                language=example["language"],
                label=label,
                label_id=LABEL_IDS[label],
                audio_path=audio_path.resolve().as_posix(),
                source_duration_s=float(example["decoded_duration_s"]),
                pause_index=pause_index,
                silence_start_s=float(pause["start"]),
                silence_end_s=float(pause["end"]),
                decision_time_s=decision_time,
                window_start_s=max(0.0, decision_time - window_seconds),
                window_end_s=decision_time,
                left_padding_s=max(0.0, window_seconds - decision_time),
                target_sample_rate_hz=16000,
                target_channels=1,
                # Deliberately hidden: Pipecat has no transcripts or context.
                model_transcript=None,
                model_dialogue_context=None,
                synthetic=None,
                provenance=payload["dataset"],
            )


def write_jsonl(path: Path, decisions: list[CanonicalDecision]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as stream:
        for decision in decisions:
            stream.write(json.dumps(asdict(decision), ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def write_database(path: Path, decisions: list[CanonicalDecision]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.execute(
            """
            CREATE TABLE decisions (
                schema_version INTEGER NOT NULL,
                decision_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                source_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                split TEXT NOT NULL CHECK(split IN ('train', 'evaluation')),
                language TEXT NOT NULL,
                label TEXT NOT NULL CHECK(label IN ('hold', 'end')),
                label_id INTEGER NOT NULL CHECK(label_id IN (0, 1)),
                audio_path TEXT NOT NULL,
                source_duration_s REAL NOT NULL,
                pause_index INTEGER,
                silence_start_s REAL,
                silence_end_s REAL,
                decision_time_s REAL NOT NULL,
                window_start_s REAL NOT NULL,
                window_end_s REAL NOT NULL,
                left_padding_s REAL NOT NULL,
                target_sample_rate_hz INTEGER NOT NULL,
                target_channels INTEGER NOT NULL,
                model_transcript TEXT,
                model_dialogue_context TEXT,
                synthetic INTEGER,
                provenance TEXT NOT NULL
            )
            """
        )
        connection.execute("CREATE INDEX decisions_split_label ON decisions(split, label)")
        connection.execute("CREATE INDEX decisions_group ON decisions(group_id)")
        columns = list(asdict(decisions[0]))
        placeholders = ",".join("?" for _ in columns)
        connection.executemany(
            f"INSERT INTO decisions ({','.join(columns)}) VALUES ({placeholders})",
            ([asdict(item)[column] for column in columns] for item in decisions),
        )
        connection.commit()
    finally:
        connection.close()
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.window_seconds <= 0 or args.score_point <= 0:
        raise ValueError("window and score-point values must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    train = list(pipecat_decisions(args.pipecat_root, args.window_seconds))
    evaluation = list(
        livekit_decisions(args.livekit_root, args.window_seconds, args.score_point)
    )
    write_jsonl(args.output / "train.jsonl", train)
    write_jsonl(args.output / "evaluation.jsonl", evaluation)
    write_database(args.output / "dataset.sqlite3", train + evaluation)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "window_seconds": args.window_seconds,
        "livekit_score_point_seconds": args.score_point,
        "model_modalities": ["audio"],
        "train": {
            "records": len(train),
            "hold": sum(item.label == "hold" for item in train),
            "end": sum(item.label == "end" for item in train),
            "groups": len({item.group_id for item in train}),
        },
        "evaluation": {
            "records": len(evaluation),
            "hold": sum(item.label == "hold" for item in evaluation),
            "end": sum(item.label == "end" for item in evaluation),
            "audio_clips": len({item.source_id for item in evaluation}),
            "groups": len({item.group_id for item in evaluation}),
        },
        "fairness": {
            "transcripts_exposed_to_model": False,
            "dialogue_context_exposed_to_model": False,
            "livekit_future_audio_exposed": False,
            "target_sample_rate_hz": 16000,
            "target_channels": 1,
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
