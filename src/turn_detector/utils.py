"""Shared reproducibility helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(*command: str) -> str | None:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_value(*args: str) -> str | None:
    return command_output("git", *args)
