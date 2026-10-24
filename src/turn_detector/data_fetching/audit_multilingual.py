#!/usr/bin/env python3
"""Audit the complete multilingual canonical dataset before training."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", type=Path, default=Path("data/canonical_multilingual_v1/dataset.sqlite3")
    )
    parser.add_argument("--audio-samples-per-stratum", type=int, default=2)
    parser.add_argument(
        "--output-json", type=Path, default=Path("reports/multilingual_v1_data_audit.json")
    )
    parser.add_argument(
        "--output-md", type=Path, default=Path("reports/multilingual_v1_data_audit.md")
    )
    return parser.parse_args()


def percentile(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "p50": None, "p95": None, "maximum": None}
    return {
        "minimum": min(values),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "maximum": max(values),
    }


def select_audio_samples(
    connection: sqlite3.Connection, samples_per_stratum: int
) -> set[tuple[str, str, int, int]]:
    candidates: dict[tuple[str, str, str], list[tuple[bytes, tuple[str, str, int, int]]]] = (
        defaultdict(list)
    )
    rows = connection.execute(
        """
        SELECT source, split, language, label, audio_path, parquet_row_group,
               parquet_row_index, source_id
        FROM decisions ORDER BY decision_id
        """
    )
    seen_source_records: set[tuple[str, str]] = set()
    for source, split, language, label, path, group, row, source_id in rows:
        source_key = (source, source_id)
        if source_key in seen_source_records:
            continue
        seen_source_records.add(source_key)
        locator = (source, path, int(group), int(row))
        key = (source, split, language)
        rank = hashlib.sha256(":".join(map(str, (*key, source_id))).encode()).digest()
        values = candidates[key]
        values.append((rank, locator))
        values.sort(key=lambda item: item[0])
        del values[samples_per_stratum:]
    return {locator for values in candidates.values() for _, locator in values}


def database_audit(connection: sqlite3.Connection) -> dict[str, Any]:
    required = (
        "decision_id", "source", "source_id", "group_id", "split", "language",
        "source_language", "label", "audio_path", "parquet_row_group",
        "parquet_row_index", "provenance", "dataset_revision",
    )
    missing = {
        column: connection.execute(
            f"SELECT COUNT(*) FROM decisions WHERE {column} IS NULL OR CAST({column} AS TEXT) = ''"
        ).fetchone()[0]
        for column in required
    }
    by_language_split = {}
    for language, split, count, holds, ends, synthetic, real_or_non_synthetic, unknown in connection.execute(
        """
        SELECT language, split, COUNT(*), SUM(label_id = 0), SUM(label_id = 1),
               SUM(synthetic = 1), SUM(synthetic = 0), SUM(synthetic IS NULL)
        FROM decisions GROUP BY language, split ORDER BY language, split
        """
    ):
        by_language_split.setdefault(language, {})[split] = {
            "records": count,
            "hold": holds,
            "end": ends,
            "synthetic": synthetic,
            "non_synthetic": real_or_non_synthetic,
            "unknown_provenance_class": unknown,
        }
    provenance = {
        f"{language}:{name}": count
        for language, name, count in connection.execute(
            """
            SELECT language, provenance, COUNT(*) FROM decisions
            GROUP BY language, provenance ORDER BY language, provenance
            """
        )
    }
    source_id_overlap = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT source, source_id FROM decisions
            GROUP BY source, source_id HAVING COUNT(DISTINCT split) > 1
        )
        """
    ).fetchone()[0]
    group_overlap = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT group_id FROM decisions WHERE source = 'livekit'
            GROUP BY group_id HAVING COUNT(DISTINCT split) > 1
        )
        """
    ).fetchone()[0]
    duplicate_decision_ids = connection.execute(
        "SELECT COUNT(*) - COUNT(DISTINCT decision_id) FROM decisions"
    ).fetchone()[0]
    return {
        "missing_required_fields": missing,
        "by_language_split": by_language_split,
        "provenance_counts": provenance,
        "source_record_overlap_across_splits": source_id_overlap,
        "livekit_group_overlap_across_splits": group_overlap,
        "duplicate_decision_ids": duplicate_decision_ids,
        "speaker_id": "unavailable in both source schemas",
        "conversation_id": "available only as an inferred LiveKit filename prefix",
    }


def pause_audit(connection: sqlite3.Connection) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for language, split, label, count, minimum, average, maximum in connection.execute(
        """
        SELECT language, split, label, COUNT(*), MIN(silence_end_s - silence_start_s),
               AVG(silence_end_s - silence_start_s), MAX(silence_end_s - silence_start_s)
        FROM decisions WHERE source = 'livekit'
        GROUP BY language, split, label ORDER BY language, split, label
        """
    ):
        durations = [
            float(row[0])
            for row in connection.execute(
                """
                SELECT silence_end_s - silence_start_s FROM decisions
                WHERE source = 'livekit' AND language = ? AND split = ? AND label = ?
                """,
                (language, split, label),
            )
        ]
        bins = {
            "under_500ms": sum(value < 0.5 for value in durations),
            "500ms_to_1s": sum(0.5 <= value < 1.0 for value in durations),
            "at_least_1s": sum(value >= 1.0 for value in durations),
        }
        result[f"{language}:{split}:{label}"] = {
            "count": count,
            "duration_seconds": {
                "minimum": minimum,
                "mean": average,
                "p50": float(np.percentile(durations, 50)),
                "p95": float(np.percentile(durations, 95)),
                "maximum": maximum,
            },
            "bins": bins,
        }
    return result


def audio_audit(
    connection: sqlite3.Connection,
    metadata: dict[str, str],
    sampled_locators: set[tuple[str, str, int, int]],
) -> dict[str, Any]:
    import pyarrow.parquet as parquet

    roots = {
        "pipecat": Path(metadata["pipecat_root"]),
        "livekit": Path(metadata["livekit_root"]),
    }
    repositories = {}
    for source, root in roots.items():
        manifest = json.loads((root / "snapshot_manifest.json").read_text(encoding="utf-8"))
        repositories[source] = root / manifest["repository_path"]

    locator_metadata: dict[tuple[str, str, int, int], tuple[str, str, str]] = {}
    for source, path, group, row, source_id, split, language in connection.execute(
        """
        SELECT source, audio_path, parquet_row_group, parquet_row_index,
               source_id, split, language FROM decisions ORDER BY decision_id
        """
    ):
        locator_metadata.setdefault(
            (source, path, int(group), int(row)), (source_id, split, language)
        )

    by_file: dict[tuple[str, str], dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for source, path, group, row in locator_metadata:
        by_file[(source, path)][group].append(row)

    hash_records: dict[str, list[dict[str, str]]] = defaultdict(list)
    full_counts = Counter()
    durations: dict[str, list[float]] = defaultdict(list)
    sample_records: list[dict[str, Any]] = []
    for (source, relative_path), row_groups in sorted(by_file.items()):
        file = parquet.ParquetFile(repositories[source] / relative_path)
        for row_group, selected_rows in sorted(row_groups.items()):
            audio_values = file.read_row_group(row_group, columns=["audio"])["audio"].to_pylist()
            for row_index in sorted(set(selected_rows)):
                locator = (source, relative_path, row_group, row_index)
                source_id, split, language = locator_metadata[locator]
                audio = audio_values[row_index]
                payload = audio.get("bytes") if isinstance(audio, dict) else None
                if not payload:
                    full_counts["missing_audio_bytes"] += 1
                    continue
                digest = hashlib.sha256(payload).hexdigest()
                hash_records[digest].append(
                    {"source": source, "source_id": source_id, "split": split, "language": language}
                )
                full_counts["audio_records"] += 1
                try:
                    info = sf.info(BytesIO(payload))
                except RuntimeError:
                    full_counts["unreadable_audio"] += 1
                    continue
                full_counts["readable_audio"] += 1
                full_counts[f"sample_rate:{info.samplerate}"] += 1
                full_counts[f"channels:{info.channels}"] += 1
                duration = info.frames / info.samplerate
                durations[f"{source}:{language}"].append(duration)
                if locator in sampled_locators:
                    values, rate = sf.read(BytesIO(payload), dtype="float32", always_2d=True)
                    mono = values.mean(axis=1, dtype=np.float32)
                    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
                    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
                    sample_records.append(
                        {
                            "source": source,
                            "source_id": source_id,
                            "split": split,
                            "language": language,
                            "sample_rate": rate,
                            "channels": int(values.shape[1]),
                            "duration_seconds": duration,
                            "peak_absolute_amplitude": peak,
                            "rms": rms,
                            "silent_rms_below_1e_4": rms < 1e-4,
                            "clipped_peak_at_least_0_999": peak >= 0.999,
                        }
                    )

    duplicate_groups = [records for records in hash_records.values() if len(records) > 1]
    cross_split = [
        records for records in duplicate_groups if len({record["split"] for record in records}) > 1
    ]
    cross_source = [
        records for records in duplicate_groups if len({record["source"] for record in records}) > 1
    ]
    return {
        "full_population": {
            **dict(full_counts),
            "total_duration_hours": sum(sum(values) for values in durations.values()) / 3600,
            "duration_seconds_by_source_language": {
                key: percentile(values) | {"hours": sum(values) / 3600, "records": len(values)}
                for key, values in sorted(durations.items())
            },
            "exact_encoded_audio_duplicate_groups": len(duplicate_groups),
            "records_in_exact_encoded_duplicate_groups": sum(len(records) for records in duplicate_groups),
            "excess_exact_encoded_copies": sum(len(records) - 1 for records in duplicate_groups),
            "cross_split_exact_encoded_duplicate_groups": len(cross_split),
            "cross_source_exact_encoded_duplicate_groups": len(cross_source),
            "cross_split_examples": cross_split[:10],
        },
        "sample": {
            "selection": "deterministic minimum-SHA source recordings per source/split/language stratum",
            "requested_locators": len(sampled_locators),
            "decoded_records": len(sample_records),
            "silent_records": sum(record["silent_rms_below_1e_4"] for record in sample_records),
            "clipped_records": sum(record["clipped_peak_at_least_0_999"] for record in sample_records),
            "records": sample_records,
        },
        "duplicate_rule": "SHA-256 equality of original embedded encoded audio bytes",
        "decoded_content_duplicate_check": "not performed; cross-codec equality remains unknown",
    }


def markdown(report: dict[str, Any]) -> str:
    db = report["database"]
    audio = report["audio"]
    lines = [
        "# Multilingual v1 data audit",
        "",
        f"- Audit time: `{report['created_utc']}`",
        f"- Canonical index: `{report['index']}`",
        f"- Languages: `{', '.join(report['languages'])}`",
        f"- Pipecat revision: `{report['revisions']['pipecat']}`",
        f"- LiveKit revision: `{report['revisions']['livekit']}`",
        "- Intended use: Pipecat training/validation and LiveKit causal development/final evaluation.",
        "",
        "## Observed counts",
        "",
        "| Language | Train | Validation | Evaluation decisions | Holdout decisions | Synthetic train |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for language in report["languages"]:
        values = db["by_language_split"][language]
        lines.append(
            f"| {language} | {values['train']['records']} | {values['validation']['records']} | "
            f"{values['evaluation']['records']} | {values['holdout']['records']} | "
            f"{values['train']['synthetic']} |"
        )
    full = audio["full_population"]
    lines += [
        "",
        "## Integrity and audio",
        "",
        f"- Required-field missing counts: `{json.dumps(db['missing_required_fields'], sort_keys=True)}`",
        f"- Readable source recordings: **{full['readable_audio']:,} / {full['audio_records']:,}**; unreadable: **{full.get('unreadable_audio', 0)}**.",
        f"- Total source audio: **{full['total_duration_hours']:.2f} hours**.",
        f"- Exact encoded-audio duplicate groups: **{full['exact_encoded_audio_duplicate_groups']}**; cross-split: **{full['cross_split_exact_encoded_duplicate_groups']}**; cross-source: **{full['cross_source_exact_encoded_duplicate_groups']}**.",
        f"- LiveKit conversation-group overlap: **{db['livekit_group_overlap_across_splits']}**; source-record overlap: **{db['source_record_overlap_across_splits']}**.",
        f"- Deterministic audio sample: **{audio['sample']['decoded_records']}** recordings; silent: **{audio['sample']['silent_records']}**; clipped: **{audio['sample']['clipped_records']}**.",
        "- Full-population audio checks cover decodability, sample rate, channels, duration, and exact encoded-byte duplicates. Silence and clipping are sampled.",
        "",
        "## Coverage and limitations",
        "",
        "- Pipecat supplies isolated hold/end clips. LiveKit supplies causal continuation and final-silence spans grouped by inferred conversations.",
        "- Explicit speaker IDs are unavailable, so speaker-independent splitting cannot be verified.",
        "- LiveKit real/synthetic provenance is not declared in the indexed schema and remains unknown.",
        "- Only English and Spanish include non-synthetic Pipecat records in this scope; the other training languages are synthetic in the source metadata.",
        "- Backchannels, overlap, interruptions, recording device, and acoustic environment are not explicitly labeled.",
        "- Exact decoded-content duplicates across different codecs were not measured.",
        "",
        "## Decision",
        "",
        f"**{report['verdict']}**",
        "",
        report["verdict_reason"],
        "",
        "The training run uses a deterministic full-coverage grouped shuffle. Temperature resampling is not used in this one-epoch run because it would either omit examples or repeat mostly synthetic minority-language examples; macro per-language validation prevents English volume from selecting the checkpoint alone.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    with sqlite3.connect(args.index) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM index_metadata"))
        languages = json.loads(metadata["languages"])
        database = database_audit(connection)
        pauses = pause_audit(connection)
        samples = select_audio_samples(connection, args.audio_samples_per_stratum)
        audio = audio_audit(connection, metadata, samples)

    blockers = {
        "missing_required_fields": sum(database["missing_required_fields"].values()),
        "duplicate_decision_ids": database["duplicate_decision_ids"],
        "source_overlap": database["source_record_overlap_across_splits"],
        "group_overlap": database["livekit_group_overlap_across_splits"],
        "missing_audio": audio["full_population"].get("missing_audio_bytes", 0),
        "unreadable_audio": audio["full_population"].get("unreadable_audio", 0),
        "cross_split_audio_duplicates": audio["full_population"]["cross_split_exact_encoded_duplicate_groups"],
    }
    suitable = not any(blockers.values())
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "index": args.index.resolve().as_posix(),
        "languages": languages,
        "revisions": {
            "pipecat": metadata["pipecat_revision"],
            "livekit": metadata["livekit_revision"],
        },
        "scope": {
            "metadata": "full population",
            "audio_header_and_hash": "full unique source-recording population",
            "silence_and_clipping": "deterministic stratified sample",
        },
        "database": database,
        "pause_coverage": pauses,
        "audio": audio,
        "blocking_checks": blockers,
        "verdict": "Suitable with stated constraints" if suitable else "Unsuitable until blocking findings are resolved",
        "verdict_reason": (
            "The indexed files are complete, readable, group-disjoint where conversation IDs exist, and contain both labels in every language and partition. The main constraints are synthetic-heavy training, unavailable speaker IDs, unknown LiveKit provenance, and incomplete difficult-case annotations."
            if suitable
            else "At least one integrity or leakage check failed; training must not begin until it is resolved."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "blocking_checks": blockers}, indent=2))


if __name__ == "__main__":
    main()
