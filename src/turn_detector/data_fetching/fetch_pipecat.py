#!/usr/bin/env python3
"""Download and verify the complete revision-pinned Pipecat training snapshot."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


DATASET = "pipecat-ai/smart-turn-data-v3.2-train"
SPLIT = "train"
DEFAULT_REVISION = "f8c95d284595defb6b88a715c356cbc5c31b7ad1"
DEFAULT_OUTPUT = Path("data/pipecat_smart_turn_v3_2_full")


def snapshot_plan(revision: str) -> dict[str, Any]:
    """Resolve the immutable revision and original Parquet shard inventory."""
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "dataset fetching requires the packages in requirements-data.txt"
        ) from error

    info = HfApi().dataset_info(DATASET, revision=revision, files_metadata=True)
    shards = []
    for sibling in info.siblings:
        if sibling.rfilename.startswith("data/") and sibling.rfilename.endswith(".parquet"):
            lfs = getattr(sibling, "lfs", None)
            shards.append(
                {
                    "path": sibling.rfilename,
                    "bytes": int(sibling.size or 0),
                    "sha256": getattr(lfs, "sha256", None),
                }
            )
    if not shards:
        raise RuntimeError(f"no Parquet shards found for {DATASET}@{revision}")
    return {
        "dataset": DATASET,
        "revision": info.sha,
        "requested_revision": revision,
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
    """Download all original shards at one immutable revision."""
    plan = snapshot_plan(revision)
    required = plan["total_bytes"]
    output.parent.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(output.parent).free
    reserve = int(minimum_free_gb * 1024**3)
    print(
        f"dataset revision: {plan['revision']}\n"
        f"shards: {plan['shard_count']}\n"
        f"download size: {required / 1024**3:.2f} GiB\n"
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
            allow_patterns=["data/*.parquet", "README.md", "LICENSE"],
            cache_dir=output / "hf_cache",
            max_workers=workers,
        )
    )

    missing = [
        shard["path"] for shard in plan["shards"] if not (repository / shard["path"]).is_file()
    ]
    size_mismatches = [
        shard["path"]
        for shard in plan["shards"]
        if (repository / shard["path"]).is_file()
        and (repository / shard["path"]).stat().st_size != shard["bytes"]
    ]
    if missing or size_mismatches:
        raise RuntimeError(
            f"snapshot verification failed: {len(missing)} missing, "
            f"{len(size_mismatches)} size mismatches"
        )

    plan["repository_path"] = os.path.relpath(repository, output)
    manifest_part.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    os.replace(manifest_part, output / "snapshot_manifest.json")
    print(f"complete dataset ready: {repository}")
    return repository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="immutable Hugging Face commit",
    )
    parser.add_argument(
        "--minimum-free-gb",
        type=float,
        default=5.0,
        help="disk space that must remain after a new download",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and print the download plan without downloading",
    )
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
