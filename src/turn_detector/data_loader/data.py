"""Read-only waveform loaders backed by the canonical dataset index."""

from __future__ import annotations

from io import BytesIO
from dataclasses import dataclass
from functools import lru_cache
import json
import numpy as np
from pathlib import Path
import random
import sqlite3
from typing import Any

try:
    from torch.utils.data import Dataset, IterableDataset
except ImportError:  # Allows metadata inspection without the training runtime.
    class Dataset:  # type: ignore[no-redef]
        pass

    class IterableDataset:  # type: ignore[no-redef]
        pass


def decode_audio_window_array(
    audio_source: bytes | Path,
    *,
    end_seconds: float | None,
    window_seconds: float,
    sample_rate: int,
    normalize_waveform: bool = False,
):
    try:
        import soundfile as sf
    except ImportError as error:
        raise RuntimeError("waveform loading requires soundfile") from error

    source = BytesIO(audio_source) if isinstance(audio_source, bytes) else audio_source
    audio, decoded_rate = sf.read(source, dtype="float32", always_2d=True)
    if decoded_rate != sample_rate:
        raise ValueError(f"expected {sample_rate} Hz, got {decoded_rate} Hz")
    waveform = np.mean(audio, axis=1, dtype=np.float32)
    if end_seconds is not None:
        waveform = waveform[: min(waveform.size, round(end_seconds * sample_rate))]
    target_samples = round(window_seconds * sample_rate)
    waveform = waveform[-target_samples:]
    if normalize_waveform and waveform.size:
        waveform = (waveform - waveform.mean()) / np.sqrt(waveform.var() + 1e-7)
    if waveform.size < target_samples:
        waveform = np.pad(waveform, (target_samples - waveform.size, 0))
    return waveform.astype(np.float32, copy=False)


def decode_audio_window(
    audio_source: bytes | Path,
    *,
    end_seconds: float | None,
    window_seconds: float,
    sample_rate: int,
    normalize_waveform: bool = False,
):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("tensor loading requires torch") from error
    return torch.from_numpy(
        decode_audio_window_array(
            audio_source,
            end_seconds=end_seconds,
            window_seconds=window_seconds,
            sample_rate=sample_rate,
            normalize_waveform=normalize_waveform,
        )
    )


def _index_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    return dict(connection.execute("SELECT key, value FROM index_metadata"))


class CanonicalPipecatDataset(IterableDataset):
    """Stream one canonical Pipecat split without materializing its audio."""

    COLUMNS = (
        "audio",
        "id",
        "language",
        "endpoint_bool",
        "synthetic",
        "dataset",
    )

    def __init__(
        self,
        index_path: str | Path,
        *,
        split: str,
        language: str = "eng",
        window_seconds: float = 8.0,
        sample_rate: int = 16000,
        shuffle: bool | None = None,
        shuffle_buffer: int = 2048,
        seed: int = 1337,
        max_examples: int | None = None,
        normalize_waveform: bool = False,
    ):
        super().__init__()
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        self.index_path = Path(index_path).resolve()
        self.split = split
        self.language = language
        self.window_seconds = window_seconds
        self.sample_rate = sample_rate
        self.shuffle = split == "train" if shuffle is None else shuffle
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.max_examples = max_examples
        self.normalize_waveform = normalize_waveform
        self.epoch = 0
        with sqlite3.connect(self.index_path) as connection:
            metadata = _index_metadata(connection)
            self._count = connection.execute(
                """
                SELECT COUNT(*) FROM decisions
                WHERE source = 'pipecat' AND split = ? AND language = ?
                """,
                (split, language),
            ).fetchone()[0]
        self.pipecat_root = Path(metadata["pipecat_root"])
        manifest = json.loads(
            (self.pipecat_root / "snapshot_manifest.json").read_text(encoding="utf-8")
        )
        self.repository = self.pipecat_root / manifest["repository_path"]

    def __len__(self) -> int:
        return min(self._count, self.max_examples or self._count)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _tasks(self) -> list[tuple[str, int, tuple[int, ...]]]:
        with sqlite3.connect(self.index_path) as connection:
            rows = connection.execute(
                """
                SELECT audio_path, parquet_row_group, parquet_row_index
                FROM decisions
                WHERE source = 'pipecat' AND split = ? AND language = ?
                ORDER BY audio_path, parquet_row_group, parquet_row_index
                """,
                (self.split, self.language),
            )
            grouped: dict[tuple[str, int], list[int]] = {}
            for path, row_group, row_index in rows:
                grouped.setdefault((path, row_group), []).append(row_index)
        tasks = [(path, group, tuple(indices)) for (path, group), indices in grouped.items()]
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(tasks)
        return tasks

    @staticmethod
    def _worker_partition(tasks):
        try:
            import torch.distributed as distributed
            from torch.utils.data import get_worker_info
        except ImportError:
            return tasks
        rank = distributed.get_rank() if distributed.is_available() and distributed.is_initialized() else 0
        world_size = distributed.get_world_size() if distributed.is_available() and distributed.is_initialized() else 1
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        partition_id = rank * worker_count + worker_id
        partitions = world_size * worker_count
        return tasks[partition_id::partitions]

    def _examples(self):
        try:
            import pyarrow.parquet as parquet
            import torch
        except ImportError as error:
            raise RuntimeError(
                "canonical training requires requirements-train.txt"
            ) from error

        yielded = 0
        rng = random.Random(self.seed + self.epoch)
        buffer: list[dict[str, Any]] = []
        for relative_path, row_group, selected_indices in self._worker_partition(self._tasks()):
            table = parquet.ParquetFile(self.repository / relative_path).read_row_group(
                row_group, columns=list(self.COLUMNS)
            )
            rows = table.to_pylist()
            indices = list(selected_indices)
            if self.shuffle:
                rng.shuffle(indices)
            for row_index in indices:
                row = rows[row_index]
                audio = row["audio"]
                audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
                if not audio_bytes:
                    raise RuntimeError(f"row {row['id']} has no embedded audio bytes")
                item = {
                    "id": row["id"],
                    "waveform": decode_audio_window(
                        audio_bytes,
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
                buffer_limit = min(self.shuffle_buffer, self.max_examples or self.shuffle_buffer)
                if self.shuffle and buffer_limit > 1:
                    buffer.append(item)
                    if len(buffer) >= buffer_limit:
                        yield buffer.pop(rng.randrange(len(buffer)))
                        yielded += 1
                else:
                    yield item
                    yielded += 1
                if self.max_examples is not None and yielded >= self.max_examples:
                    return
        while buffer:
            yield buffer.pop(rng.randrange(len(buffer)))
            yielded += 1
            if self.max_examples is not None and yielded >= self.max_examples:
                return

    def __iter__(self):
        return iter(self._examples())


@dataclass(frozen=True)
class EvaluationDecision:
    decision_id: str
    source_id: str
    group_id: str
    evaluation_partition: str
    audio_storage: str
    audio_path: str
    parquet_row_group: int | None
    parquet_row_index: int | None
    label: str
    label_id: int
    pause_index: int
    silence_start_s: float
    silence_end_s: float
    decision_time_s: float | None
    source_duration_s: float
    provenance: str


class CanonicalEvaluationDataset(Dataset):
    """Load external decisions with audio truncated at each scoring timestamp."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        language: str = "eng",
        partition: str = "ENG_evaluation_dev",
        window_seconds: float = 8.0,
        sample_rate: int = 16000,
        normalize_waveform: bool = False,
    ):
        self.index_path = Path(index_path).resolve()
        self.window_seconds = window_seconds
        self.sample_rate = sample_rate
        self.partition = partition
        self.normalize_waveform = normalize_waveform
        with sqlite3.connect(self.index_path) as connection:
            metadata = _index_metadata(connection)
            rows = connection.execute(
                """
                SELECT decision_id, source_id, group_id, evaluation_partition,
                       audio_storage, audio_path, parquet_row_group, parquet_row_index,
                       label, label_id, pause_index, silence_start_s, silence_end_s,
                       decision_time_s, source_duration_s, provenance
                FROM decisions
                WHERE split = 'evaluation' AND language = ?
                  AND evaluation_partition = ?
                ORDER BY source_id, pause_index
                """,
                (language, partition),
            ).fetchall()
        self.root = Path(metadata["livekit_root"])
        manifest = json.loads(
            (self.root / "snapshot_manifest.json").read_text(encoding="utf-8")
        )
        self.repository = self.root / manifest["repository_path"]
        self.decisions = tuple(
            EvaluationDecision(
                decision_id=row[0],
                source_id=row[1],
                group_id=row[2],
                evaluation_partition=row[3],
                audio_storage=row[4],
                audio_path=row[5],
                parquet_row_group=row[6],
                parquet_row_index=row[7],
                label=row[8],
                label_id=row[9],
                pause_index=row[10],
                silence_start_s=row[11],
                silence_end_s=row[12],
                decision_time_s=row[13],
                source_duration_s=row[14],
                provenance=row[15],
            )
            for row in rows
        )

    def __len__(self) -> int:
        return len(self.decisions)

    @lru_cache(maxsize=4)
    def _parquet_audio(self, relative_path: str, row_group: int):
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise RuntimeError("evaluation loading requires requirements-data.txt") from error
        return parquet.ParquetFile(self.repository / relative_path).read_row_group(
            row_group, columns=["audio"]
        )["audio"].to_pylist()

    def _audio_source(self, decision: EvaluationDecision) -> bytes | Path:
        if decision.audio_storage == "file":
            return self.root / decision.audio_path
        if decision.parquet_row_group is None or decision.parquet_row_index is None:
            raise ValueError(f"{decision.decision_id}: missing Parquet locator")
        audio = self._parquet_audio(decision.audio_path, decision.parquet_row_group)[
            decision.parquet_row_index
        ]
        audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
        if not audio_bytes:
            raise RuntimeError(f"{decision.decision_id}: no embedded audio bytes")
        return audio_bytes

    def waveform_at(self, index: int, decision_time_s: float):
        decision = self.decisions[index]
        if decision_time_s < 0 or decision_time_s > decision.source_duration_s + 1e-6:
            raise ValueError("decision timestamp is outside the available audio")
        return decode_audio_window(
            self._audio_source(decision),
            end_seconds=decision_time_s,
            window_seconds=self.window_seconds,
            sample_rate=self.sample_rate,
            normalize_waveform=self.normalize_waveform,
        )

    def __getitem__(self, index: int):
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("evaluation loading requires torch") from error
        decision = self.decisions[index]
        return {
            "decision_id": decision.decision_id,
            "source_id": decision.source_id,
            "group_id": decision.group_id,
            "evaluation_partition": decision.evaluation_partition,
            "waveform": self.waveform_at(index, decision.silence_end_s),
            "label": torch.tensor(decision.label_id, dtype=torch.float32),
            "label_name": decision.label,
            "pause_index": decision.pause_index,
            "silence_start_s": decision.silence_start_s,
            "silence_end_s": decision.silence_end_s,
            "decision_time_s": decision.decision_time_s,
            "source_duration_s": decision.source_duration_s,
            "provenance": decision.provenance,
        }
