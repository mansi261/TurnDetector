"""Audit exact audio duplicates across canonical training splits."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/canonical/dataset.sqlite3"))
    parser.add_argument("--language", default="eng")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/audits/english_split_duplicate_audit.json"),
    )
    parser.add_argument("--fail-on-cross-split", action="store_true")
    return parser.parse_args()


def audit(index_path: Path, language: str) -> dict:
    import pyarrow.parquet as parquet

    with sqlite3.connect(index_path) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM index_metadata"))
        rows = connection.execute(
            """
            SELECT audio_path, parquet_row_group, parquet_row_index,
                   source_id, split, label
            FROM decisions
            WHERE source = 'pipecat' AND language = ?
            ORDER BY audio_path, parquet_row_group, parquet_row_index
            """,
            (language,),
        ).fetchall()
    pipecat_root = Path(metadata["pipecat_root"])
    manifest = json.loads(
        (pipecat_root / "snapshot_manifest.json").read_text(encoding="utf-8")
    )
    repository_root = pipecat_root / manifest["repository_path"]
    tasks: dict[tuple[str, int], list[tuple[int, str, str, str]]] = defaultdict(list)
    for path, row_group, row_index, source_id, split, label in rows:
        tasks[(path, row_group)].append((row_index, source_id, split, label))

    hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
    for task_number, ((path, row_group), selected) in enumerate(tasks.items(), 1):
        values = parquet.ParquetFile(repository_root / path).read_row_group(
            row_group, columns=["audio"]
        )["audio"].to_pylist()
        for row_index, source_id, split, label in selected:
            audio = values[row_index]
            audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
            if not audio_bytes:
                raise RuntimeError(f"{source_id} has no embedded audio")
            digest = hashlib.sha256(audio_bytes).hexdigest()
            hashes[digest].append(
                {"source_id": source_id, "split": split, "label": label}
            )
        if task_number % 100 == 0:
            print(f"scanned {task_number}/{len(tasks)} row groups")

    duplicate_groups = [items for items in hashes.values() if len(items) > 1]
    cross_split = [
        items for items in duplicate_groups if len({item["split"] for item in items}) > 1
    ]
    conflicting_labels = [
        items for items in duplicate_groups if len({item["label"] for item in items}) > 1
    ]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "language": language,
        "records_scanned": len(rows),
        "unique_audio_hashes": len(hashes),
        "exact_duplicate_groups": len(duplicate_groups),
        "records_in_exact_duplicate_groups": sum(len(items) for items in duplicate_groups),
        "cross_split_duplicate_groups": len(cross_split),
        "cross_split_duplicate_records": sum(len(items) for items in cross_split),
        "conflicting_label_duplicate_groups": len(conflicting_labels),
        "cross_split_examples": cross_split[:100],
        "method": "SHA-256 over embedded encoded audio bytes",
        "limitation": "This detects byte-identical audio, not perceptually similar recordings.",
    }


def main() -> None:
    args = parse_args()
    report = audit(args.index, args.language)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.fail_on_cross_split and report["cross_split_duplicate_groups"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
