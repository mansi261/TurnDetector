#!/usr/bin/env python3
"""Build a canonical metadata index over training and evaluation data.

Audio remains embedded in immutable Parquet snapshots. The index stores stable
row locators, normalized labels, group-disjoint splits, and queryable provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 3
LABEL_IDS = {"hold": 0, "end": 1}
EVALUATION_PARTITIONS = (
    "ENG_evaluation_dev",
    "ENG_holdout_a",
    "ENG_holdout_b",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipecat-root",
        type=Path,
        default=Path("data/pipecat_smart_turn_v3_2_full"),
    )
    parser.add_argument(
        "--livekit-root",
        type=Path,
        default=Path("data/livekit_eot_bench_ENG_public_400"),
    )
    parser.add_argument(
        "--exposed-livekit-root",
        type=Path,
        default=Path("data/LiveKit_50_English_Evaluation_Examples"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/canonical"))
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", default="turn-detector-v1")
    parser.add_argument("--evaluation-split-seed", default="livekit-eng-evaluation-v1")
    parser.add_argument("--window-seconds", type=float, default=8.0)
    return parser.parse_args()


def deterministic_split(record_id: str, seed: str, validation_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{record_id}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return "validation" if fraction < validation_fraction else "train"


def canonical_language(language: str) -> str:
    """Use ISO 639-3 codes consistently across sources."""
    return {"en": "eng"}.get(language, language)


def livekit_group_id(source_id: str) -> str:
    marker = "__user_turn_"
    return source_id.split(marker, 1)[0] if marker in source_id else source_id


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = DELETE;
        PRAGMA synchronous = NORMAL;

        CREATE TABLE index_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE decisions (
            schema_version INTEGER NOT NULL,
            decision_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            source_id TEXT NOT NULL,
            group_id TEXT NOT NULL,
            split TEXT NOT NULL CHECK(split IN ('train', 'validation', 'evaluation')),
            evaluation_partition TEXT CHECK(
                evaluation_partition IN (
                    'ENG_evaluation_dev', 'ENG_holdout_a', 'ENG_holdout_b'
                ) OR evaluation_partition IS NULL
            ),
            language TEXT NOT NULL,
            label TEXT NOT NULL CHECK(label IN ('hold', 'end')),
            label_id INTEGER NOT NULL CHECK(label_id IN (0, 1)),
            audio_storage TEXT NOT NULL CHECK(audio_storage IN ('parquet', 'file')),
            audio_path TEXT NOT NULL,
            parquet_row_group INTEGER,
            parquet_row_index INTEGER,
            source_duration_s REAL,
            pause_index INTEGER,
            silence_start_s REAL,
            silence_end_s REAL,
            decision_time_s REAL,
            window_start_s REAL,
            window_end_s REAL,
            left_padding_s REAL,
            target_sample_rate_hz INTEGER NOT NULL,
            target_channels INTEGER NOT NULL,
            model_transcript TEXT,
            model_dialogue_context TEXT,
            synthetic INTEGER CHECK(synthetic IN (0, 1) OR synthetic IS NULL),
            midfiller INTEGER CHECK(midfiller IN (0, 1) OR midfiller IS NULL),
            endfiller INTEGER CHECK(endfiller IN (0, 1) OR endfiller IS NULL),
            provenance TEXT NOT NULL,
            dataset_revision TEXT
        );

        CREATE INDEX decisions_language_idx
            ON decisions(language);
        CREATE INDEX decisions_split_language_label_idx
            ON decisions(split, language, label_id);
        CREATE INDEX decisions_evaluation_partition_idx
            ON decisions(evaluation_partition, language, label_id);
        CREATE INDEX decisions_source_locator_idx
            ON decisions(source, audio_path, parquet_row_group, parquet_row_index);
        CREATE INDEX decisions_provenance_idx
            ON decisions(provenance);
        """
    )


DECISION_COLUMNS = (
    "schema_version",
    "decision_id",
    "source",
    "source_id",
    "group_id",
    "split",
    "evaluation_partition",
    "language",
    "label",
    "label_id",
    "audio_storage",
    "audio_path",
    "parquet_row_group",
    "parquet_row_index",
    "source_duration_s",
    "pause_index",
    "silence_start_s",
    "silence_end_s",
    "decision_time_s",
    "window_start_s",
    "window_end_s",
    "left_padding_s",
    "target_sample_rate_hz",
    "target_channels",
    "model_transcript",
    "model_dialogue_context",
    "synthetic",
    "midfiller",
    "endfiller",
    "provenance",
    "dataset_revision",
)


def insert_rows(connection: sqlite3.Connection, rows: Iterable[tuple[Any, ...]]) -> None:
    placeholders = ",".join("?" for _ in DECISION_COLUMNS)
    connection.executemany(
        f"INSERT INTO decisions ({','.join(DECISION_COLUMNS)}) VALUES ({placeholders})",
        rows,
    )


def index_pipecat(
    connection: sqlite3.Connection,
    root: Path,
    *,
    split_seed: str,
    validation_fraction: float,
) -> tuple[dict[str, Any], int]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "canonical indexing requires the packages in requirements-data.txt"
        ) from error

    manifest_path = root / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repository = root / manifest["repository_path"]
    total = 0
    columns = [
        "id",
        "language",
        "endpoint_bool",
        "synthetic",
        "midfiller",
        "endfiller",
        "dataset",
    ]

    for shard_number, shard in enumerate(manifest["shards"], 1):
        relative_path = shard["path"]
        shard_path = repository / relative_path
        parquet_file = parquet.ParquetFile(shard_path)
        for row_group in range(parquet_file.num_row_groups):
            values = parquet_file.read_row_group(row_group, columns=columns).to_pydict()
            rows = []
            for row_index, record_id in enumerate(values["id"]):
                endpoint = bool(values["endpoint_bool"][row_index])
                label = "end" if endpoint else "hold"
                rows.append(
                    (
                        SCHEMA_VERSION,
                        f"pipecat:{record_id}",
                        "pipecat",
                        record_id,
                        record_id,
                        deterministic_split(record_id, split_seed, validation_fraction),
                        None,
                        canonical_language(values["language"][row_index]),
                        label,
                        LABEL_IDS[label],
                        "parquet",
                        relative_path,
                        row_group,
                        row_index,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        16000,
                        1,
                        None,
                        None,
                        int(bool(values["synthetic"][row_index])),
                        _nullable_bool(values["midfiller"][row_index]),
                        _nullable_bool(values["endfiller"][row_index]),
                        values["dataset"][row_index],
                        manifest["revision"],
                    )
                )
            insert_rows(connection, rows)
            total += len(rows)
        print(f"[{shard_number:>2}/{len(manifest['shards'])}] indexed {relative_path}")
    return manifest, total


def _nullable_bool(value: Any) -> int | None:
    return None if value is None else int(bool(value))


def exposed_livekit_ids(root: Path) -> set[str]:
    path = root / "examples.json"
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {example["id"] for example in payload["examples"]}


def assign_evaluation_partitions(
    records: list[dict[str, Any]], exposed_ids: set[str], seed: str
) -> tuple[dict[str, str], dict[str, Any]]:
    grouped: dict[str, list[str]] = {}
    for record in records:
        grouped.setdefault(record["group_id"], []).append(record["id"])
    exposed_groups = {
        group_id
        for group_id, source_ids in grouped.items()
        if any(source_id in exposed_ids for source_id in source_ids)
    }
    targets = {
        "ENG_evaluation_dev": round(len(records) * 0.50),
        "ENG_holdout_a": round(len(records) * 0.25),
    }
    targets["ENG_holdout_b"] = len(records) - sum(targets.values())
    assignments = {group_id: "ENG_evaluation_dev" for group_id in exposed_groups}
    counts = {partition: 0 for partition in EVALUATION_PARTITIONS}
    counts["ENG_evaluation_dev"] = sum(len(grouped[group_id]) for group_id in exposed_groups)
    remaining = sorted(
        (group_id for group_id in grouped if group_id not in exposed_groups),
        key=lambda group_id: hashlib.sha256(f"{seed}:{group_id}".encode()).digest(),
    )
    for group_id in remaining:
        partition = max(
            EVALUATION_PARTITIONS,
            key=lambda name: (
                (targets[name] - counts[name]) / targets[name],
                hashlib.sha256(f"{seed}:{group_id}:{name}".encode()).digest(),
            ),
        )
        assignments[group_id] = partition
        counts[partition] += len(grouped[group_id])
    manifest = {
        "language": "eng",
        "language_label": "ENG",
        "seed": seed,
        "unit": "complete user turn",
        "group_key": "source ID prefix before __user_turn_",
        "policy": "50/25/25 by turns with previously exposed groups forced to development",
        "targets": targets,
        "turn_counts": counts,
        "exposed_turn_ids": len(exposed_ids),
        "exposed_conversation_groups": len(exposed_groups),
        "group_assignments": dict(sorted(assignments.items())),
    }
    return assignments, manifest


def index_livekit(
    connection: sqlite3.Connection,
    root: Path,
    *,
    exposed_root: Path,
    split_seed: str,
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "canonical indexing requires the packages in requirements-data.txt"
        ) from error

    manifest = json.loads((root / "snapshot_manifest.json").read_text(encoding="utf-8"))
    repository = root / manifest["repository_path"]
    records = []
    for shard in manifest["shards"]:
        relative_path = shard["path"]
        parquet_file = parquet.ParquetFile(repository / relative_path)
        for row_group in range(parquet_file.num_row_groups):
            values = parquet_file.read_row_group(
                row_group,
                columns=["id", "language", "duration", "silence_spans"],
            ).to_pylist()
            for row_index, value in enumerate(values):
                if value["language"] != "en":
                    raise ValueError(f"{value['id']}: expected English, got {value['language']}")
                if not value["silence_spans"]:
                    raise ValueError(f"{value['id']}: no silence spans")
                records.append(
                    {
                        **value,
                        "group_id": livekit_group_id(value["id"]),
                        "audio_path": relative_path,
                        "parquet_row_group": row_group,
                        "parquet_row_index": row_index,
                    }
                )

    exposed_ids = exposed_livekit_ids(exposed_root)
    source_ids = {record["id"] for record in records}
    unknown_exposed = exposed_ids - source_ids
    if unknown_exposed:
        raise ValueError(f"{len(unknown_exposed)} exposed IDs are absent from the pinned snapshot")
    assignments, partition_manifest = assign_evaluation_partitions(
        records, exposed_ids, split_seed
    )
    rows = []
    for record in records:
        spans = record["silence_spans"]
        for pause_index, span in enumerate(spans):
            label = "end" if pause_index == len(spans) - 1 else "hold"
            silence_start = float(span["start"])
            silence_end = float(span["end"])
            rows.append(
                (
                    SCHEMA_VERSION,
                    f"livekit:{record['id']}:pause_{pause_index:02d}",
                    "livekit",
                    record["id"],
                    record["group_id"],
                    "evaluation",
                    assignments[record["group_id"]],
                    "eng",
                    label,
                    LABEL_IDS[label],
                    "parquet",
                    record["audio_path"],
                    record["parquet_row_group"],
                    record["parquet_row_index"],
                    float(record["duration"]),
                    pause_index,
                    silence_start,
                    silence_end,
                    None,
                    None,
                    None,
                    None,
                    16000,
                    1,
                    None,
                    None,
                    None,
                    None,
                    None,
                    manifest["dataset"],
                    manifest["revision"],
                )
            )
    insert_rows(connection, rows)
    partition_manifest["dataset"] = manifest["dataset"]
    partition_manifest["dataset_revision"] = manifest["revision"]
    partition_manifest["decision_counts"] = {
        partition: connection.execute(
            "SELECT COUNT(*) FROM decisions WHERE evaluation_partition = ?",
            (partition,),
        ).fetchone()[0]
        for partition in EVALUATION_PARTITIONS
    }
    return manifest, len(rows), partition_manifest


def build_summary(connection: sqlite3.Connection, metadata: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_revision": metadata["revision"],
        "model_modalities": ["audio"],
        "splits": {},
        "evaluation_partitions": {},
        "languages": {},
        "fairness": {
            "transcripts_exposed_to_model": False,
            "dialogue_context_exposed_to_model": False,
            "future_evaluation_audio_exposed": False,
            "target_sample_rate_hz": 16000,
            "target_channels": 1,
        },
    }
    for split, count, hold, end in connection.execute(
        """
        SELECT split, COUNT(*),
               SUM(CASE WHEN label = 'hold' THEN 1 ELSE 0 END),
               SUM(CASE WHEN label = 'end' THEN 1 ELSE 0 END)
        FROM decisions GROUP BY split ORDER BY split
        """
    ):
        summary["splits"][split] = {"records": count, "hold": hold, "end": end}
    for language, count in connection.execute(
        """
        SELECT language, COUNT(*) FROM decisions
        WHERE source = 'pipecat' GROUP BY language ORDER BY language
        """
    ):
        summary["languages"][language] = count
    for partition, decisions, turns, groups, hold, end in connection.execute(
        """
        SELECT evaluation_partition, COUNT(*), COUNT(DISTINCT source_id),
               COUNT(DISTINCT group_id),
               SUM(CASE WHEN label = 'hold' THEN 1 ELSE 0 END),
               SUM(CASE WHEN label = 'end' THEN 1 ELSE 0 END)
        FROM decisions WHERE evaluation_partition IS NOT NULL
        GROUP BY evaluation_partition ORDER BY evaluation_partition
        """
    ):
        summary["evaluation_partitions"][partition] = {
            "decisions": decisions,
            "turns": turns,
            "conversation_groups": groups,
            "hold": hold,
            "end": end,
        }
    return summary


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between zero and one")
    if args.window_seconds <= 0:
        raise ValueError("window seconds must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    database = args.output / "dataset.sqlite3"
    temporary = database.with_suffix(database.suffix + ".part")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        create_schema(connection)
        manifest, train_records = index_pipecat(
            connection,
            args.pipecat_root,
            split_seed=args.split_seed,
            validation_fraction=args.validation_fraction,
        )
        livekit_manifest, evaluation_records, partition_manifest = index_livekit(
            connection,
            args.livekit_root,
            exposed_root=args.exposed_livekit_root,
            split_seed=args.evaluation_split_seed,
        )
        index_metadata = {
            "schema_version": str(SCHEMA_VERSION),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "pipecat_dataset": manifest["dataset"],
            "pipecat_revision": manifest["revision"],
            "pipecat_root": args.pipecat_root.resolve().as_posix(),
            "livekit_root": args.livekit_root.resolve().as_posix(),
            "livekit_dataset": livekit_manifest["dataset"],
            "livekit_revision": livekit_manifest["revision"],
            "livekit_language": "eng",
            "livekit_language_label": "ENG",
            "validation_fraction": str(args.validation_fraction),
            "split_seed": args.split_seed,
            "evaluation_split_seed": args.evaluation_split_seed,
            "window_seconds": str(args.window_seconds),
            "evaluation_decision_timing": "full silence spans; no fixed decision timestamp",
        }
        connection.executemany(
            "INSERT INTO index_metadata(key, value) VALUES (?, ?)",
            index_metadata.items(),
        )
        summary = build_summary(connection, manifest)
        connection.commit()
    finally:
        connection.close()

    os.replace(temporary, database)
    summary_path = args.output / "summary.json"
    summary_part = summary_path.with_suffix(".json.part")
    summary_part.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(summary_part, summary_path)
    partition_path = args.output / "ENG_evaluation_partitions.json"
    partition_part = partition_path.with_suffix(".json.part")
    partition_part.write_text(
        json.dumps(partition_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partition_part, partition_path)
    print(
        f"canonical index ready: {database} "
        f"({train_records} training-source records, {evaluation_records} evaluation decisions)"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
