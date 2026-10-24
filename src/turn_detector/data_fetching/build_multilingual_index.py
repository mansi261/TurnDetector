#!/usr/bin/env python3
"""Build the isolated 14-language canonical dataset index."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterable

from turn_detector.data_fetching.build_canonical_index import deterministic_split
from turn_detector.data_fetching.fetch_livekit import LANGUAGE_CODES


SCHEMA_VERSION = 4
LANGUAGES = tuple(LANGUAGE_CODES.values())
SOURCE_CONFIGS = {value: key for key, value in LANGUAGE_CODES.items()}
LABEL_IDS = {"hold": 0, "end": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipecat-root", type=Path, default=Path("data/pipecat_smart_turn_v3_2_full")
    )
    parser.add_argument(
        "--livekit-root", type=Path, default=Path("data/livekit_eot_bench_MULTI_public")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/canonical_multilingual_v1")
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--training-split-seed", default="multilingual-training-v1")
    parser.add_argument("--external-split-seed", default="multilingual-external-v1")
    parser.add_argument("--evaluation-fraction", type=float, default=0.75)
    parser.add_argument("--window-seconds", type=float, default=8.0)
    return parser.parse_args()


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
            source TEXT NOT NULL CHECK(source IN ('pipecat', 'livekit')),
            source_id TEXT NOT NULL,
            group_id TEXT NOT NULL,
            split TEXT NOT NULL CHECK(split IN ('train', 'validation', 'evaluation', 'holdout')),
            evaluation_partition TEXT CHECK(
                evaluation_partition IN ('MULTI_evaluation', 'MULTI_holdout')
                OR evaluation_partition IS NULL
            ),
            language TEXT NOT NULL,
            source_language TEXT NOT NULL,
            label TEXT NOT NULL CHECK(label IN ('hold', 'end')),
            label_id INTEGER NOT NULL CHECK(label_id IN (0, 1)),
            audio_storage TEXT NOT NULL CHECK(audio_storage = 'parquet'),
            audio_path TEXT NOT NULL,
            parquet_row_group INTEGER NOT NULL,
            parquet_row_index INTEGER NOT NULL,
            source_duration_s REAL,
            pause_index INTEGER,
            silence_start_s REAL,
            silence_end_s REAL,
            target_sample_rate_hz INTEGER NOT NULL,
            target_channels INTEGER NOT NULL,
            synthetic INTEGER CHECK(synthetic IN (0, 1) OR synthetic IS NULL),
            midfiller INTEGER CHECK(midfiller IN (0, 1) OR midfiller IS NULL),
            endfiller INTEGER CHECK(endfiller IN (0, 1) OR endfiller IS NULL),
            provenance TEXT NOT NULL,
            dataset_revision TEXT NOT NULL
        );

        CREATE INDEX decisions_language_idx ON decisions(language);
        CREATE INDEX decisions_split_language_label_idx
            ON decisions(split, language, label_id);
        CREATE INDEX decisions_partition_language_label_idx
            ON decisions(evaluation_partition, language, label_id);
        CREATE INDEX decisions_group_split_idx ON decisions(group_id, split);
        CREATE INDEX decisions_source_locator_idx
            ON decisions(source, audio_path, parquet_row_group, parquet_row_index);
        CREATE INDEX decisions_provenance_idx ON decisions(provenance);
        """
    )


COLUMNS = (
    "schema_version", "decision_id", "source", "source_id", "group_id", "split",
    "evaluation_partition", "language", "source_language", "label", "label_id",
    "audio_storage", "audio_path", "parquet_row_group", "parquet_row_index",
    "source_duration_s", "pause_index", "silence_start_s", "silence_end_s",
    "target_sample_rate_hz", "target_channels", "synthetic", "midfiller",
    "endfiller", "provenance", "dataset_revision",
)


def insert_rows(connection: sqlite3.Connection, rows: Iterable[tuple[Any, ...]]) -> None:
    placeholders = ",".join("?" for _ in COLUMNS)
    connection.executemany(
        f"INSERT INTO decisions ({','.join(COLUMNS)}) VALUES ({placeholders})", rows
    )


def nullable_bool(value: Any) -> int | None:
    return None if value is None else int(bool(value))


def livekit_group_id(language: str, source_id: str) -> str:
    marker = "__user_turn_"
    prefix = source_id.split(marker, 1)[0] if marker in source_id else source_id
    return f"{language}:{prefix}"


def index_pipecat(
    connection: sqlite3.Connection,
    root: Path,
    *,
    split_seed: str,
    validation_fraction: float,
) -> tuple[dict[str, Any], int]:
    import pyarrow.parquet as parquet

    manifest = json.loads((root / "snapshot_manifest.json").read_text(encoding="utf-8"))
    repository = root / manifest["repository_path"]
    wanted = set(LANGUAGES)
    total = 0
    columns = [
        "id", "language", "endpoint_bool", "synthetic", "midfiller", "endfiller",
        "dataset",
    ]
    for shard_number, shard in enumerate(manifest["shards"], 1):
        relative_path = shard["path"]
        file = parquet.ParquetFile(repository / relative_path)
        for row_group in range(file.num_row_groups):
            values = file.read_row_group(row_group, columns=columns).to_pydict()
            rows = []
            for row_index, record_id in enumerate(values["id"]):
                language = values["language"][row_index]
                if language not in wanted:
                    continue
                label = "end" if bool(values["endpoint_bool"][row_index]) else "hold"
                rows.append(
                    (
                        SCHEMA_VERSION, f"pipecat:{record_id}", "pipecat", record_id,
                        record_id,
                        deterministic_split(record_id, split_seed, validation_fraction),
                        None, language, language, label, LABEL_IDS[label], "parquet",
                        relative_path, row_group, row_index, None, None, None, None,
                        16000, 1, int(bool(values["synthetic"][row_index])),
                        nullable_bool(values["midfiller"][row_index]),
                        nullable_bool(values["endfiller"][row_index]),
                        values["dataset"][row_index], manifest["revision"],
                    )
                )
            insert_rows(connection, rows)
            total += len(rows)
        print(f"[{shard_number:>2}/{len(manifest['shards'])}] indexed {relative_path}")
    return manifest, total


def assign_livekit_groups(
    records: list[dict[str, Any]], seed: str, evaluation_fraction: float
) -> tuple[dict[str, str], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["group_id"]].append(record)

    assignments: dict[str, str] = {}
    language_summary: dict[str, Any] = {}
    for language in LANGUAGES:
        language_groups = {
            group: items
            for group, items in grouped.items()
            if items[0]["language"] == language
        }
        target = round(
            sum(len(items) for items in language_groups.values()) * evaluation_fraction
        )
        ordered = sorted(
            language_groups,
            key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).digest(),
        )
        counts = {"MULTI_evaluation": 0, "MULTI_holdout": 0}
        for group in ordered:
            group_size = len(language_groups[group])
            evaluation_gap = target - counts["MULTI_evaluation"]
            partition = (
                "MULTI_evaluation"
                if evaluation_gap > 0
                and abs(evaluation_gap - group_size) <= abs(evaluation_gap)
                else "MULTI_holdout"
            )
            assignments[group] = partition
            counts[partition] += group_size
        language_summary[language] = {
            "turns": sum(len(items) for items in language_groups.values()),
            "conversation_groups": len(language_groups),
            "target_evaluation_turns": target,
            "turn_counts": counts,
        }
    return assignments, language_summary


def index_livekit(
    connection: sqlite3.Connection,
    root: Path,
    *,
    split_seed: str,
    evaluation_fraction: float,
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    import pyarrow.parquet as parquet

    manifest = json.loads((root / "snapshot_manifest.json").read_text(encoding="utf-8"))
    if set(manifest["languages"]) != set(LANGUAGES):
        raise ValueError("LiveKit manifest does not contain the exact 14-language scope")
    repository = root / manifest["repository_path"]
    records: list[dict[str, Any]] = []
    for shard in manifest["shards"]:
        relative_path = shard["path"]
        expected_source_language = relative_path.split("/")[1]
        expected_language = LANGUAGE_CODES[expected_source_language]
        file = parquet.ParquetFile(repository / relative_path)
        for row_group in range(file.num_row_groups):
            values = file.read_row_group(
                row_group, columns=["id", "language", "duration", "silence_spans"]
            ).to_pylist()
            for row_index, value in enumerate(values):
                if value["language"] != expected_source_language:
                    raise ValueError(
                        f"{value['id']}: expected {expected_source_language}, "
                        f"got {value['language']}"
                    )
                if not value["silence_spans"]:
                    raise ValueError(f"{value['id']}: no silence spans")
                records.append(
                    {
                        **value,
                        "language": expected_language,
                        "source_language": expected_source_language,
                        "group_id": livekit_group_id(expected_language, value["id"]),
                        "audio_path": relative_path,
                        "parquet_row_group": row_group,
                        "parquet_row_index": row_index,
                    }
                )

    assignments, language_summary = assign_livekit_groups(
        records, split_seed, evaluation_fraction
    )
    rows = []
    for record in records:
        partition = assignments[record["group_id"]]
        split = "evaluation" if partition == "MULTI_evaluation" else "holdout"
        spans = record["silence_spans"]
        for pause_index, span in enumerate(spans):
            label = "end" if pause_index == len(spans) - 1 else "hold"
            rows.append(
                (
                    SCHEMA_VERSION,
                    f"livekit:{record['language']}:{record['id']}:pause_{pause_index:02d}",
                    "livekit", record["id"], record["group_id"], split, partition,
                    record["language"], record["source_language"], label,
                    LABEL_IDS[label], "parquet", record["audio_path"],
                    record["parquet_row_group"], record["parquet_row_index"],
                    float(record["duration"]), pause_index, float(span["start"]),
                    float(span["end"]), 16000, 1, None, None, None,
                    manifest["dataset"], manifest["revision"],
                )
            )
    insert_rows(connection, rows)
    split_manifest = {
        "dataset": manifest["dataset"],
        "dataset_revision": manifest["revision"],
        "seed": split_seed,
        "policy": "75/25 by turns within language with conversation groups kept intact",
        "group_key": "canonical language plus source ID prefix before __user_turn_",
        "evaluation_fraction": evaluation_fraction,
        "languages": language_summary,
        "group_assignments": dict(sorted(assignments.items())),
    }
    return manifest, len(rows), split_manifest


def decision_hash(connection: sqlite3.Connection, split: str) -> str:
    digest = hashlib.sha256()
    for (decision_id,) in connection.execute(
        "SELECT decision_id FROM decisions WHERE split = ? ORDER BY decision_id", (split,)
    ):
        digest.update(decision_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def build_summary(connection: sqlite3.Connection) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "languages": list(LANGUAGES),
        "partitions": {},
        "partition_decision_id_sha256": {},
        "by_language": {},
        "fairness": {
            "transcripts_exposed_to_model": False,
            "dialogue_context_exposed_to_model": False,
            "future_audio_exposed_to_model": False,
            "silence_spans_used_by_scorer_only": True,
            "target_sample_rate_hz": 16000,
            "target_channels": 1,
        },
    }
    for split, count, holds, ends in connection.execute(
        """
        SELECT split, COUNT(*), SUM(label_id = 0), SUM(label_id = 1)
        FROM decisions GROUP BY split ORDER BY split
        """
    ):
        summary["partitions"][split] = {"records": count, "hold": holds, "end": ends}
        summary["partition_decision_id_sha256"][split] = decision_hash(connection, split)
    for language in LANGUAGES:
        partitions = {}
        for split, count, holds, ends, turns, groups in connection.execute(
            """
            SELECT split, COUNT(*), SUM(label_id = 0), SUM(label_id = 1),
                   COUNT(DISTINCT source_id), COUNT(DISTINCT group_id)
            FROM decisions WHERE language = ? GROUP BY split ORDER BY split
            """,
            (language,),
        ):
            partitions[split] = {
                "records": count, "hold": holds, "end": ends,
                "source_records": turns, "groups": groups,
            }
        summary["by_language"][language] = partitions
    return summary


def validate_index(connection: sqlite3.Connection) -> None:
    observed = {
        row[0]
        for row in connection.execute("SELECT DISTINCT language FROM decisions")
    }
    if observed != set(LANGUAGES):
        raise RuntimeError(f"language coverage differs: {sorted(observed)}")
    for split in ("train", "validation", "evaluation", "holdout"):
        coverage = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT language FROM decisions WHERE split = ?", (split,)
            )
        }
        if coverage != set(LANGUAGES):
            raise RuntimeError(f"{split} language coverage differs: {sorted(coverage)}")
    overlap = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT group_id FROM decisions WHERE source = 'livekit'
            GROUP BY group_id HAVING COUNT(DISTINCT split) > 1
        )
        """
    ).fetchone()[0]
    if overlap:
        raise RuntimeError(f"{overlap} LiveKit conversation groups cross external splits")
    duplicates = connection.execute(
        "SELECT COUNT(*) - COUNT(DISTINCT decision_id) FROM decisions"
    ).fetchone()[0]
    if duplicates:
        raise RuntimeError(f"{duplicates} duplicate decision IDs")


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between zero and one")
    if not 0 < args.evaluation_fraction < 1:
        raise ValueError("--evaluation-fraction must be between zero and one")
    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    database = args.output / "dataset.sqlite3"
    temporary = database.with_suffix(".sqlite3.part")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        create_schema(connection)
        pipecat_manifest, pipecat_records = index_pipecat(
            connection, args.pipecat_root, split_seed=args.training_split_seed,
            validation_fraction=args.validation_fraction,
        )
        livekit_manifest, livekit_decisions, split_manifest = index_livekit(
            connection, args.livekit_root, split_seed=args.external_split_seed,
            evaluation_fraction=args.evaluation_fraction,
        )
        metadata = {
            "schema_version": str(SCHEMA_VERSION),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "languages": json.dumps(LANGUAGES),
            "pipecat_dataset": pipecat_manifest["dataset"],
            "pipecat_revision": pipecat_manifest["revision"],
            "pipecat_root": args.pipecat_root.resolve().as_posix(),
            "livekit_dataset": livekit_manifest["dataset"],
            "livekit_revision": livekit_manifest["revision"],
            "livekit_root": args.livekit_root.resolve().as_posix(),
            "training_split_seed": args.training_split_seed,
            "validation_fraction": str(args.validation_fraction),
            "external_split_seed": args.external_split_seed,
            "evaluation_fraction": str(args.evaluation_fraction),
            "window_seconds": str(args.window_seconds),
        }
        connection.executemany(
            "INSERT INTO index_metadata(key, value) VALUES (?, ?)", metadata.items()
        )
        validate_index(connection)
        summary = build_summary(connection)
        connection.commit()
    finally:
        connection.close()

    os.replace(temporary, database)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "MULTI_partitions.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"multilingual canonical index ready: {database} "
        f"({pipecat_records} Pipecat records, {livekit_decisions} LiveKit decisions)"
    )
    print(json.dumps(summary["partitions"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
