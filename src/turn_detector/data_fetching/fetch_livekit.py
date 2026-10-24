#!/usr/bin/env python3
"""Download and verify revision-pinned public LiveKit language samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


DATASET = "livekit/eot-bench-data"
DEFAULT_LANGUAGE_CONFIGS = ("en",)
LANGUAGE_CODES = {
    "ar": "ara",
    "de": "deu",
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "hi": "hin",
    "id": "ind",
    "it": "ita",
    "ja": "jpn",
    "ko": "kor",
    "nl": "nld",
    "pt": "por",
    "tr": "tur",
    "zh": "zho",
}
SPLIT = "validation"
DEFAULT_REVISION = "ca9d98a9686b920a2d8c9eb984224ba9be74e4dd"
DEFAULT_OUTPUT = Path("data/livekit_eot_bench_ENG_public_400")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_plan(revision: str, language_configs: tuple[str, ...]) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "dataset fetching requires the packages in requirements-data.txt"
        ) from error

    info = HfApi().dataset_info(DATASET, revision=revision, files_metadata=True)
    shards = []
    for sibling in info.siblings:
        if any(
            sibling.rfilename.startswith(f"data/{language}/")
            for language in language_configs
        ) and sibling.rfilename.endswith(".parquet"):
            lfs = getattr(sibling, "lfs", None)
            shards.append(
                {
                    "path": sibling.rfilename,
                    "bytes": int(sibling.size or 0),
                    "sha256": getattr(lfs, "sha256", None),
                }
            )
    found_configs = {shard["path"].split("/")[1] for shard in shards}
    missing_configs = sorted(set(language_configs) - found_configs)
    if missing_configs:
        raise RuntimeError(
            f"no Parquet shards found for configs {missing_configs} in {DATASET}@{revision}"
        )
    canonical_languages = [LANGUAGE_CODES[language] for language in language_configs]
    plan = {
        "dataset": DATASET,
        "revision": info.sha,
        "requested_revision": revision,
        "source_configs": list(language_configs),
        "languages": canonical_languages,
        "scope": "Complete public evaluation samples for the selected languages",
        "split": SPLIT,
        "license": "CC BY 4.0",
        "storage": "original Parquet shards with embedded audio",
        "shard_count": len(shards),
        "total_bytes": sum(shard["bytes"] for shard in shards),
        "shards": sorted(shards, key=lambda shard: shard["path"]),
    }
    if len(language_configs) == 1:
        language = language_configs[0]
        plan.update(
            {
                "source_config": language,
                "language": LANGUAGE_CODES[language],
                "language_label": LANGUAGE_CODES[language].upper(),
                "scope": "Complete public English evaluation sample (400 turns)",
            }
        )
    return plan


def fetch_snapshot(
    output: Path,
    *,
    revision: str,
    workers: int,
    minimum_free_gb: float,
    dry_run: bool,
    language_configs: tuple[str, ...] = DEFAULT_LANGUAGE_CONFIGS,
) -> Path | None:
    plan = snapshot_plan(revision, language_configs)
    required = plan["total_bytes"]
    output.parent.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(output.parent).free
    reserve = int(minimum_free_gb * 1024**3)
    print(
        f"dataset revision: {plan['revision']}\n"
        f"languages: {', '.join(plan['languages'])}\n"
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
            allow_patterns=[
                *(f"data/{language}/*.parquet" for language in language_configs),
                "README.md",
            ],
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
    print(f"complete public language samples ready: {repository}")
    return repository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--minimum-free-gb", type=float, default=2.0)
    parser.add_argument(
        "--languages",
        default=",".join(DEFAULT_LANGUAGE_CONFIGS),
        help="comma-separated LiveKit language configurations",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    language_configs = tuple(
        language.strip() for language in args.languages.split(",") if language.strip()
    )
    unknown = sorted(set(language_configs) - set(LANGUAGE_CODES))
    if not language_configs or unknown:
        raise ValueError(f"unsupported language configurations: {unknown}")
    fetch_snapshot(
        args.output,
        revision=args.revision,
        workers=args.workers,
        minimum_free_gb=args.minimum_free_gb,
        dry_run=args.dry_run,
        language_configs=language_configs,
    )


if __name__ == "__main__":
    main()
