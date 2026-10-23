#!/usr/bin/env python3
"""Download and verify the complete public English LiveKit sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


DATASET = "livekit/eot-bench-data"
LANGUAGE_CONFIG = "en"
CANONICAL_LANGUAGE = "eng"
DISPLAY_LANGUAGE = "ENG"
SPLIT = "validation"
DEFAULT_REVISION = "ca9d98a9686b920a2d8c9eb984224ba9be74e4dd"
DEFAULT_OUTPUT = Path("data/livekit_eot_bench_ENG_public_400")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_plan(revision: str) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "dataset fetching requires the packages in requirements-data.txt"
        ) from error

    info = HfApi().dataset_info(DATASET, revision=revision, files_metadata=True)
    prefix = f"data/{LANGUAGE_CONFIG}/"
    shards = []
    for sibling in info.siblings:
        if sibling.rfilename.startswith(prefix) and sibling.rfilename.endswith(".parquet"):
            lfs = getattr(sibling, "lfs", None)
            shards.append(
                {
                    "path": sibling.rfilename,
                    "bytes": int(sibling.size or 0),
                    "sha256": getattr(lfs, "sha256", None),
                }
            )
    if not shards:
        raise RuntimeError(f"no English Parquet shards found for {DATASET}@{revision}")
    return {
        "dataset": DATASET,
        "revision": info.sha,
        "requested_revision": revision,
        "source_config": LANGUAGE_CONFIG,
        "language": CANONICAL_LANGUAGE,
        "language_label": DISPLAY_LANGUAGE,
        "scope": "Complete public English evaluation sample (400 turns)",
        "split": SPLIT,
        "license": "CC BY 4.0",
        "storage": "original Parquet shards with embedded audio",
        "shard_count": len(shards),
        "total_bytes": sum(shard["bytes"] for shard in shards),
        "shards": sorted(shards, key=lambda shard: shard["path"]),
    }


def fetch_snapshot(
    output: Path,
    *,
    revision: str,
    workers: int,
    minimum_free_gb: float,
    dry_run: bool,
) -> Path | None:
    plan = snapshot_plan(revision)
    required = plan["total_bytes"]
    output.parent.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(output.parent).free
    reserve = int(minimum_free_gb * 1024**3)
    print(
        f"dataset revision: {plan['revision']}\n"
        f"language: {plan['language_label']} ({plan['language']})\n"
        f"shards: {plan['shard_count']}\n"
        f"download size: {required / 1024**2:.2f} MiB\n"
        f"free disk: {available / 1024**3:.2f} GiB"
    )
    if dry_run:
        return None
    if available < required + reserve and not (output / "snapshot_manifest.json").is_file():
        raise RuntimeError(
            "insufficient disk space: download requires "
            f"{required / 1024**3:.2f} GiB plus {minimum_free_gb:.1f} GiB reserve"
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "dataset fetching requires the packages in requirements-data.txt"
        ) from error

    output.mkdir(parents=True, exist_ok=True)
    manifest_part = output / "snapshot_manifest.json.part"
    manifest_part.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    repository = Path(
        snapshot_download(
            repo_id=DATASET,
            repo_type="dataset",
            revision=plan["revision"],
            allow_patterns=[f"data/{LANGUAGE_CONFIG}/*.parquet", "README.md"],
            cache_dir=output / "hf_cache",
            max_workers=workers,
        )
    )

    missing = []
    size_mismatches = []
    checksum_mismatches = []
    for shard in plan["shards"]:
        shard_path = repository / shard["path"]
        if not shard_path.is_file():
            missing.append(shard["path"])
            continue
        if shard_path.stat().st_size != shard["bytes"]:
            size_mismatches.append(shard["path"])
            continue
        if shard["sha256"] and sha256_file(shard_path) != shard["sha256"]:
            checksum_mismatches.append(shard["path"])
    if missing or size_mismatches or checksum_mismatches:
        raise RuntimeError(
            "snapshot verification failed: "
            f"{len(missing)} missing, {len(size_mismatches)} size mismatches, "
            f"{len(checksum_mismatches)} checksum mismatches"
        )

    plan["repository_path"] = os.path.relpath(repository, output)
    plan["verification"] = "size and SHA-256 verified"
    manifest_part.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    os.replace(manifest_part, output / "snapshot_manifest.json")
    print(f"complete public English sample ready: {repository}")
    return repository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--minimum-free-gb", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fetch_snapshot(
        args.output,
        revision=args.revision,
        workers=args.workers,
        minimum_free_gb=args.minimum_free_gb,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
