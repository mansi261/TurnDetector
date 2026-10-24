"""Load the isolated multilingual canonical dataset."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import json
import numpy as np
from pathlib import Path
import random
import sqlite3
from typing import Any, Iterable

try:
    from torch.utils.data import Dataset, IterableDataset
except ImportError:
    class Dataset:  # type: ignore[no-redef]
        pass

    class IterableDataset:  # type: ignore[no-redef]
        pass

from turn_detector.data_loader.data import decode_audio_window


def index_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    return dict(connection.execute("SELECT key, value FROM index_metadata"))


class MultilingualPipecatDataset(IterableDataset):
    """Stream a deterministic, full-coverage multilingual Pipecat split."""

    COLUMNS = ("audio", "id", "language", "endpoint_bool", "synthetic", "dataset")

    def __init__(
        self,
        index_path: str | Path,
        *,
        split: str,
        languages: Iterable[str],
        window_seconds: float = 8.0,
        sample_rate: int = 16000,
        shuffle: bool | None = None,
        seed: int = 1337,
        decision_ids: Iterable[str] | None = None,
        start_example: int = 0,
        normalize_waveform: bool = False,
    ):
        super().__init__()
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        self.index_path = Path(index_path).resolve()
        self.split = split
        self.languages = tuple(sorted(set(languages)))
        if not self.languages:
            raise ValueError("at least one language is required")
        self.window_seconds = window_seconds
        self.sample_rate = sample_rate
        self.shuffle = split == "train" if shuffle is None else shuffle
        self.seed = seed
        self.epoch = 0
        self.start_example = start_example
        self.normalize_waveform = normalize_waveform
        selected_ids = tuple(decision_ids or ())
        placeholders = ",".join("?" for _ in self.languages)
        query = f"""
            SELECT decision_id, audio_path, parquet_row_group, parquet_row_index
            FROM decisions WHERE source = 'pipecat' AND split = ?
              AND language IN ({placeholders})
        """
        parameters: list[Any] = [split, *self.languages]
        if selected_ids:
            query += f" AND decision_id IN ({','.join('?' for _ in selected_ids)})"
            parameters.extend(selected_ids)
        query += " ORDER BY audio_path, parquet_row_group, parquet_row_index"
        with sqlite3.connect(self.index_path) as connection:
            metadata = index_metadata(connection)
            rows = connection.execute(query, parameters).fetchall()
        self._rows = tuple(rows)
        if start_example < 0 or start_example > len(self._rows):
            raise ValueError("start_example is outside the dataset")
        self.pipecat_root = Path(metadata["pipecat_root"])
        manifest = json.loads(
            (self.pipecat_root / "snapshot_manifest.json").read_text(encoding="utf-8")
        )
        self.repository = self.pipecat_root / manifest["repository_path"]

    @property
    def total_examples(self) -> int:
        return len(self._rows)

    def __len__(self) -> int:
        return len(self._rows) - self.start_example

    def set_epoch(self, epoch: int, *, start_example: int = 0) -> None:
        self.epoch = epoch
        if start_example < 0 or start_example > len(self._rows):
            raise ValueError("start_example is outside the dataset")
        self.start_example = start_example

    def _tasks(self) -> list[tuple[str, int, list[int]]]:
        grouped: dict[tuple[str, int], list[int]] = {}
        for _, path, row_group, row_index in self._rows:
            grouped.setdefault((path, row_group), []).append(row_index)
        tasks = [(path, group, indices) for (path, group), indices in grouped.items()]
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(tasks)
            for _, _, indices in tasks:
                rng.shuffle(indices)
        return tasks

    def __iter__(self):
        import pyarrow.parquet as parquet
        import torch

        skipped = 0
        for relative_path, row_group, indices in self._tasks():
            if skipped + len(indices) <= self.start_example:
                skipped += len(indices)
                continue
            start = max(0, self.start_example - skipped)
            selected = indices[start:]
            skipped += len(indices)
            table = parquet.ParquetFile(self.repository / relative_path).read_row_group(
                row_group, columns=list(self.COLUMNS)
            )
            rows = table.to_pylist()
            for row_index in selected:
                row = rows[row_index]
                audio = row["audio"]
                payload = audio.get("bytes") if isinstance(audio, dict) else None
                if not payload:
                    raise RuntimeError(f"row {row['id']} has no embedded audio")
                yield {
                    "id": row["id"],
                    "waveform": decode_audio_window(
                        payload,
                        end_seconds=None,
                        window_seconds=self.window_seconds,
                        sample_rate=self.sample_rate,
                        normalize_waveform=self.normalize_waveform,
                    ),
                    "label": torch.tensor(int(row["endpoint_bool"]), dtype=torch.float32),
                    "language": row["language"],
                    "synthetic": bool(row["synthetic"]),
                    "source_dataset": row["dataset"],
                }


@dataclass(frozen=True)
class MultilingualEvaluationDecision:
    decision_id: str
    source_id: str
    group_id: str
    split: str
    evaluation_partition: str
    language: str
    audio_path: str
    parquet_row_group: int
    parquet_row_index: int
    label: str
    label_id: int
    pause_index: int
    silence_start_s: float
    silence_end_s: float
    source_duration_s: float
    provenance: str


class MultilingualEvaluationDataset(Dataset):
    """Load multilingual LiveKit decisions causally from one named partition."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        languages: Iterable[str],
        partition: str,
        window_seconds: float = 8.0,
        sample_rate: int = 16000,
        normalize_waveform: bool = False,
    ):
        if partition not in {"MULTI_evaluation", "MULTI_holdout"}:
            raise ValueError("unsupported multilingual partition")
        self.index_path = Path(index_path).resolve()
        self.languages = tuple(sorted(set(languages)))
        self.partition = partition
        self.window_seconds = window_seconds
        self.sample_rate = sample_rate
        self.normalize_waveform = normalize_waveform
        placeholders = ",".join("?" for _ in self.languages)
        with sqlite3.connect(self.index_path) as connection:
            metadata = index_metadata(connection)
            rows = connection.execute(
                f"""
                SELECT decision_id, source_id, group_id, split, evaluation_partition,
                       language, audio_path, parquet_row_group, parquet_row_index,
                       label, label_id, pause_index, silence_start_s, silence_end_s,
                       source_duration_s, provenance
                FROM decisions WHERE source = 'livekit' AND evaluation_partition = ?
                  AND language IN ({placeholders})
                ORDER BY language, source_id, pause_index
                """,
                (partition, *self.languages),
            ).fetchall()
        self.root = Path(metadata["livekit_root"])
        manifest = json.loads(
            (self.root / "snapshot_manifest.json").read_text(encoding="utf-8")
        )
        self.repository = self.root / manifest["repository_path"]
        self.decisions = tuple(MultilingualEvaluationDecision(*row) for row in rows)

    def __len__(self) -> int:
        return len(self.decisions)

    @lru_cache(maxsize=8)
    def _parquet_audio(self, relative_path: str, row_group: int):
        import pyarrow.parquet as parquet

        return parquet.ParquetFile(self.repository / relative_path).read_row_group(
            row_group, columns=["audio"]
        )["audio"].to_pylist()

    @lru_cache(maxsize=32)
    def _decoded_audio(self, relative_path: str, row_group: int, row_index: int):
        import soundfile as sf

        audio = self._parquet_audio(relative_path, row_group)[row_index]
        payload = audio.get("bytes") if isinstance(audio, dict) else None
        if not payload:
            raise RuntimeError("source recording has no embedded audio")
        values, rate = sf.read(BytesIO(payload), dtype="float32", always_2d=True)
        if rate != self.sample_rate:
            raise ValueError(f"expected {self.sample_rate} Hz, got {rate} Hz")
        return np.mean(values, axis=1, dtype=np.float32)

    def waveform_at(self, index: int, decision_time_s: float):
        import torch

        decision = self.decisions[index]
        if decision_time_s < 0 or decision_time_s > decision.source_duration_s + 1e-6:
            raise ValueError("decision timestamp is outside the available audio")
        waveform = self._decoded_audio(
            decision.audio_path, decision.parquet_row_group, decision.parquet_row_index
        )
        waveform = waveform[: min(waveform.size, round(decision_time_s * self.sample_rate))]
        target_samples = round(self.window_seconds * self.sample_rate)
        waveform = waveform[-target_samples:]
        if self.normalize_waveform and waveform.size:
            waveform = (waveform - waveform.mean()) / np.sqrt(waveform.var() + 1e-7)
        if waveform.size < target_samples:
            waveform = np.pad(waveform, (target_samples - waveform.size, 0))
        return torch.from_numpy(waveform.astype(np.float32, copy=False))

    def __getitem__(self, index: int):
        import torch

        decision = self.decisions[index]
        return {
            "decision_id": decision.decision_id,
            "source_id": decision.source_id,
            "group_id": decision.group_id,
            "partition": decision.evaluation_partition,
            "language": decision.language,
            "waveform": self.waveform_at(index, decision.silence_end_s),
            "label": torch.tensor(decision.label_id, dtype=torch.float32),
            "label_name": decision.label,
            "pause_index": decision.pause_index,
            "silence_start_s": decision.silence_start_s,
            "silence_end_s": decision.silence_end_s,
            "source_duration_s": decision.source_duration_s,
            "provenance": decision.provenance,
        }
