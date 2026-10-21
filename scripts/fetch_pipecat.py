#!/usr/bin/env python3
"""Fetch a balanced Pipecat Smart Turn sample and build a SQLite index.

The Hugging Face dataset server returns expiring audio URLs. This script always
refreshes metadata, downloads to temporary files, validates non-empty payloads,
and atomically moves successful downloads into place. Existing valid files are
left untouched, so interrupted runs can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sqlite3
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DATASET = "pipecat-ai/smart-turn-data-v3.2-train"
CONFIG = "default"
SPLIT = "train"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
USER_AGENT = "TurnDetector-data-fetcher/1.0"


def ssl_context() -> ssl.SSLContext:
    """Return a verified TLS context even when framework Python lacks CA links."""
    try:
        import certifi  # type: ignore[import-not-found]

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        system_bundle = Path("/etc/ssl/cert.pem")
        if system_bundle.is_file():
            return ssl.create_default_context(cafile=str(system_bundle))
        return ssl.create_default_context()


SSL_CONTEXT = ssl_context()


@dataclass(frozen=True)
class Example:
    sample_no: int
    hf_row_idx: int
    id: str
    language: str
    endpoint_bool: bool
    midfiller: bool | None
    endfiller: bool | None
    synthetic: bool
    spoken_text: str | None
    source_dataset: str
    audio_url: str

    @property
    def label(self) -> str:
        return "complete" if self.endpoint_bool else "incomplete"

    @property
    def filename(self) -> str:
        return f"{self.sample_no:04d}_{self.id}.flac"

    @property
    def relative_audio_path(self) -> Path:
        return Path("audio") / self.label / self.filename


def request_bytes(url: str, *, retries: int, timeout: float) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(
                request, timeout=timeout, context=SSL_CONTEXT
            ) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            last_error = error
            if attempt == retries:
                break
            time.sleep(min(2**attempt, 8) + random.random())
    raise RuntimeError(f"request failed after {retries + 1} attempts: {url}") from last_error


def fetch_rows(offset: int, length: int, *, retries: int, timeout: float) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    payload = json.loads(request_bytes(f"{ROWS_ENDPOINT}?{query}", retries=retries, timeout=timeout))
    return payload["rows"]


def select_examples(
    count: int,
    language: str,
    *,
    page_size: int,
    max_rows: int,
    retries: int,
    timeout: float,
) -> list[Example]:
    if count % 2:
        raise ValueError("--count must be even for an exactly balanced sample")
    target_per_class = count // 2
    selected: dict[bool, list[dict[str, Any]]] = {True: [], False: []}

    for offset in range(0, max_rows, page_size):
        rows = fetch_rows(offset, page_size, retries=retries, timeout=timeout)
        if not rows:
            break
        for wrapper in rows:
            row = wrapper["row"]
            label = bool(row["endpoint_bool"])
            if row["language"] == language and len(selected[label]) < target_per_class:
                selected[label].append(wrapper)
        if all(len(values) >= target_per_class for values in selected.values()):
            break

    if any(len(values) < target_per_class for values in selected.values()):
        counts = {"complete": len(selected[True]), "incomplete": len(selected[False])}
        raise RuntimeError(f"could not find a balanced sample within {max_rows} rows: {counts}")

    ordered = selected[True] + selected[False]
    examples: list[Example] = []
    for sample_no, wrapper in enumerate(ordered, 1):
        row = wrapper["row"]
        audio = row.get("audio") or []
        if not audio or not audio[0].get("src"):
            raise RuntimeError(f"row {wrapper['row_idx']} has no downloadable audio URL")
        examples.append(
            Example(
                sample_no=sample_no,
                hf_row_idx=int(wrapper["row_idx"]),
                id=row["id"],
                language=row["language"],
                endpoint_bool=bool(row["endpoint_bool"]),
                midfiller=row.get("midfiller"),
                endfiller=row.get("endfiller"),
                synthetic=bool(row["synthetic"]),
                spoken_text=row.get("spoken_text"),
                source_dataset=row["dataset"],
                audio_url=audio[0]["src"],
            )
        )
    return examples


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_one(example: Example, root: Path, *, retries: int, timeout: float) -> tuple[Example, Path]:
    destination = root / example.relative_audio_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return example, destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        payload = request_bytes(example.audio_url, retries=retries, timeout=timeout)
        if len(payload) < 4 or payload[:4] != b"fLaC":
            raise RuntimeError(f"download for {example.id} is not a valid FLAC payload")
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return example, destination


def download_all(
    examples: Iterable[Example], root: Path, *, workers: int, retries: int, timeout: float
) -> dict[str, Path]:
    examples = list(examples)
    downloaded: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_one, example, root, retries=retries, timeout=timeout): example
            for example in examples
        }
        for completed, future in enumerate(as_completed(futures), 1):
            example, path = future.result()
            downloaded[example.id] = path
            print(f"[{completed:>3}/{len(examples)}] {example.label:<10} {example.id}")
    return downloaded


def write_manifest(examples: list[Example], root: Path) -> None:
    fields = [
        "sample_no",
        "id",
        "hf_row_idx",
        "relative_audio_path",
        "language",
        "endpoint_bool",
        "label",
        "midfiller",
        "endfiller",
        "synthetic",
        "spoken_text",
        "source_dataset",
    ]
    temporary = root / "manifest.csv.part"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for example in examples:
            writer.writerow(
                {
                    "sample_no": example.sample_no,
                    "id": example.id,
                    "hf_row_idx": example.hf_row_idx,
                    "relative_audio_path": example.relative_audio_path.as_posix(),
                    "language": example.language,
                    "endpoint_bool": example.endpoint_bool,
                    "label": example.label,
                    "midfiller": example.midfiller,
                    "endfiller": example.endfiller,
                    "synthetic": example.synthetic,
                    "spoken_text": example.spoken_text,
                    "source_dataset": example.source_dataset,
                }
            )
    os.replace(temporary, root / "manifest.csv")


def build_database_from_manifest(root: Path) -> Path:
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    database = root / "dataset.sqlite3"
    temporary = root / "dataset.sqlite3.part"
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode = DELETE;
            CREATE TABLE dataset_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE examples (
                id TEXT PRIMARY KEY,
                sample_no INTEGER NOT NULL UNIQUE,
                hf_row_idx INTEGER NOT NULL,
                split TEXT NOT NULL,
                language TEXT NOT NULL,
                endpoint_bool INTEGER NOT NULL CHECK(endpoint_bool IN (0, 1)),
                label TEXT NOT NULL CHECK(label IN ('complete', 'incomplete')),
                midfiller INTEGER CHECK(midfiller IN (0, 1) OR midfiller IS NULL),
                endfiller INTEGER CHECK(endfiller IN (0, 1) OR endfiller IS NULL),
                synthetic INTEGER NOT NULL CHECK(synthetic IN (0, 1)),
                spoken_text TEXT,
                source_dataset TEXT NOT NULL,
                audio_path TEXT NOT NULL UNIQUE,
                audio_bytes INTEGER NOT NULL CHECK(audio_bytes > 0),
                audio_sha256 TEXT NOT NULL
            );
            CREATE INDEX examples_label_idx ON examples(label);
            CREATE INDEX examples_language_idx ON examples(language);
            CREATE INDEX examples_source_idx ON examples(source_dataset);
            """
        )
        metadata = {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "license": "CC BY 4.0",
            "selection": "balanced deterministic English sample",
        }
        connection.executemany(
            "INSERT INTO dataset_metadata(key, value) VALUES (?, ?)", metadata.items()
        )

        with manifest.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                audio_path = root / row["relative_audio_path"]
                if not audio_path.is_file() or audio_path.stat().st_size == 0:
                    raise RuntimeError(f"missing or empty audio file: {audio_path}")
                nullable_bool = lambda value: None if value == "" else int(value.lower() == "true")
                connection.execute(
                    """
                    INSERT INTO examples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["id"],
                        int(row["sample_no"]),
                        int(row["hf_row_idx"]),
                        SPLIT,
                        row["language"],
                        int(row["endpoint_bool"].lower() == "true"),
                        row["label"],
                        nullable_bool(row["midfiller"]),
                        nullable_bool(row["endfiller"]),
                        int(row["synthetic"].lower() == "true"),
                        row["spoken_text"] or None,
                        row["source_dataset"],
                        row["relative_audio_path"],
                        audio_path.stat().st_size,
                        sha256_file(audio_path),
                    ),
                )
        connection.commit()
        result = connection.execute(
            "SELECT COUNT(*), SUM(endpoint_bool), SUM(audio_bytes) FROM examples"
        ).fetchone()
        print(f"indexed {result[0]} examples ({result[1]} complete), {result[2]} audio bytes")
    finally:
        connection.close()
    os.replace(temporary, database)
    return database


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/pipecat_smart_turn_v3_2_sample"))
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--language", default="eng")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=100, choices=range(1, 101))
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="rebuild SQLite from an existing manifest without network access",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.index_only:
        examples = select_examples(
            args.count,
            args.language,
            page_size=args.page_size,
            max_rows=args.max_rows,
            retries=args.retries,
            timeout=args.timeout,
        )
        download_all(
            examples,
            args.output,
            workers=args.workers,
            retries=args.retries,
            timeout=args.timeout,
        )
        write_manifest(examples, args.output)
    database = build_database_from_manifest(args.output)
    print(f"database ready: {database}")


if __name__ == "__main__":
    main()
