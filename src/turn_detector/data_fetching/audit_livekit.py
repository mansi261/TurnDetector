#!/usr/bin/env python3
"""Audit the complete public English LiveKit sample."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import statistics


def group_id(source_id: str) -> str:
    marker = "__user_turn_"
    return source_id.split(marker, 1)[0] if marker in source_id else source_id


def duration_summary(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "min_s": None, "median_s": None, "p95_s": None, "max_s": None}
    return {
        "count": len(ordered),
        "min_s": ordered[0],
        "median_s": statistics.median(ordered),
        "p95_s": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max_s": ordered[-1],
    }


def pause_bins(values: list[float]) -> dict[str, int]:
    bins = Counter()
    for value in values:
        if value < 0.2:
            bins["under_200ms"] += 1
        elif value < 0.5:
            bins["200ms_to_500ms"] += 1
        elif value < 1.0:
            bins["500ms_to_1s"] += 1
        elif value < 2.0:
            bins["1s_to_2s"] += 1
        else:
            bins["2s_to_5s"] += 1
    return dict(sorted(bins.items()))


def audit(root: Path) -> dict:
    import numpy as np
    import pyarrow.parquet as parquet
    import soundfile as sf

    manifest = json.loads((root / "snapshot_manifest.json").read_text(encoding="utf-8"))
    repository = root / manifest["repository_path"]
    required = ("id", "audio", "language", "duration", "silence_spans", "words", "messages")
    records = []
    locators = []
    for shard in manifest["shards"]:
        relative_path = shard["path"]
        parquet_file = parquet.ParquetFile(repository / relative_path)
        missing_columns = sorted(set(required) - set(parquet_file.schema_arrow.names))
        if missing_columns:
            raise RuntimeError(f"{relative_path} missing columns: {missing_columns}")
        for row_group in range(parquet_file.num_row_groups):
            rows = parquet_file.read_row_group(row_group, columns=list(required)).to_pylist()
            records.extend(rows)
            locators.extend((relative_path, row_group, index) for index in range(len(rows)))

    ids = [row["id"] for row in records]
    missing = Counter()
    malformed = Counter()
    languages = Counter()
    sample_rates = Counter()
    channels = Counter()
    formats = Counter()
    audio_hashes: dict[str, list[str]] = defaultdict(list)
    decoded_hashes: dict[str, list[str]] = defaultdict(list)
    hold_durations = []
    end_durations = []
    total_audio_seconds = 0.0
    clipped_recordings = 0
    silent_recordings = 0

    for row in records:
        source_id = row["id"]
        for field in required:
            if row[field] is None:
                missing[field] += 1
        languages[str(row["language"])] += 1
        spans = row["silence_spans"] or []
        if not spans:
            malformed["empty_silence_spans"] += 1
        previous_end = -1.0
        for index, span in enumerate(spans):
            start = float(span["start"])
            end = float(span["end"])
            if start < 0 or end <= start or end > float(row["duration"]) + 0.02:
                malformed["silence_span_out_of_bounds"] += 1
            if start < previous_end - 1e-6:
                malformed["overlapping_silence_spans"] += 1
            previous_end = end
            duration = end - start
            (end_durations if index == len(spans) - 1 else hold_durations).append(duration)

        audio = row["audio"] or {}
        audio_bytes = audio.get("bytes")
        if not audio_bytes:
            malformed["missing_audio_bytes"] += 1
            continue
        audio_hashes[hashlib.sha256(audio_bytes).hexdigest()].append(source_id)
        try:
            samples, sample_rate = sf.read(BytesIO(audio_bytes), dtype="float32", always_2d=True)
            info = sf.info(BytesIO(audio_bytes))
        except Exception:
            malformed["unreadable_audio"] += 1
            continue
        sample_rates[str(sample_rate)] += 1
        channels[str(samples.shape[1])] += 1
        formats[info.format] += 1
        decoded_hashes[hashlib.sha256(np.asarray(samples, dtype="<f4").tobytes()).hexdigest()].append(
            source_id
        )
        decoded_duration = samples.shape[0] / sample_rate
        total_audio_seconds += decoded_duration
        if abs(decoded_duration - float(row["duration"])) > 0.02:
            malformed["duration_mismatch_over_20ms"] += 1
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        clipped_recordings += int(peak >= 0.999)
        silent_recordings += int(peak <= 1e-4)

    exact_groups = [items for items in audio_hashes.values() if len(items) > 1]
    decoded_groups = [items for items in decoded_hashes.values() if len(items) > 1]
    duplicate_ids = [item for item, count in Counter(ids).items() if count > 1]
    groups = [group_id(item) for item in ids]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "identity": {
            "dataset": manifest["dataset"],
            "revision": manifest["revision"],
            "source_config": manifest["source_config"],
            "canonical_language": manifest["language"],
            "language_label": manifest["language_label"],
            "source_split": manifest["split"],
            "license": manifest["license"],
            "scope": manifest["scope"],
        },
        "records": {
            "turns": len(records),
            "unique_turn_ids": len(set(ids)),
            "duplicate_turn_ids": len(duplicate_ids),
            "conversation_groups": len(set(groups)),
            "speaker_ids_available": False,
            "conversation_id_method": "ID prefix before __user_turn_",
        },
        "audio": {
            "readable": len(records) - malformed["unreadable_audio"] - malformed["missing_audio_bytes"],
            "hours": total_audio_seconds / 3600,
            "sample_rates_hz": dict(sample_rates),
            "channels": dict(channels),
            "formats": dict(formats),
            "silent_recordings": silent_recordings,
            "clipped_recordings_peak_ge_0_999": clipped_recordings,
        },
        "annotations": {
            "hold_pauses": duration_summary(hold_durations),
            "end_pauses": duration_summary(end_durations),
            "hold_pause_bins": pause_bins(hold_durations),
            "end_pause_bins": pause_bins(end_durations),
            "missing_fields": dict(missing),
            "malformed": dict(malformed),
            "languages": dict(languages),
        },
        "duplicates": {
            "encoded_audio_groups": len(exact_groups),
            "records_in_encoded_audio_groups": sum(len(items) for items in exact_groups),
            "decoded_audio_groups": len(decoded_groups),
            "records_in_decoded_audio_groups": sum(len(items) for items in decoded_groups),
            "examples": decoded_groups[:10],
        },
        "provenance": {
            "row_level_source_metadata_available": False,
            "real_synthetic_counts": {"unknown": len(records)},
        },
        "locators_scanned": len(locators),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("data/livekit_eot_bench_ENG_public_400")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/audits/livekit_ENG_public_400_audit.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
