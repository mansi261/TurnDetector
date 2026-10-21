"""Read-only loader for the locally indexed Pipecat dataset."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class PipecatExample:
    id: str
    audio_path: Path
    label: str
    endpoint_bool: bool
    language: str
    synthetic: bool
    source_dataset: str
    spoken_text: str | None


class PipecatDataset:
    """Query examples without loading audio into memory."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path).resolve()
        if not self.database_path.is_file():
            raise FileNotFoundError(self.database_path)
        self.root = self.database_path.parent

    def __len__(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM examples").fetchone()[0])

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{self.database_path.as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def iter_examples(
        self,
        *,
        label: str | None = None,
        synthetic: bool | None = None,
    ) -> Iterator[PipecatExample]:
        clauses: list[str] = []
        parameters: list[object] = []
        if label is not None:
            if label not in {"complete", "incomplete"}:
                raise ValueError("label must be 'complete' or 'incomplete'")
            clauses.append("label = ?")
            parameters.append(label)
        if synthetic is not None:
            clauses.append("synthetic = ?")
            parameters.append(int(synthetic))

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            "SELECT id, audio_path, label, endpoint_bool, language, synthetic, "
            f"source_dataset, spoken_text FROM examples{where} ORDER BY sample_no"
        )
        with self._connect() as connection:
            for row in connection.execute(query, parameters):
                yield PipecatExample(
                    id=row["id"],
                    audio_path=self.root / row["audio_path"],
                    label=row["label"],
                    endpoint_bool=bool(row["endpoint_bool"]),
                    language=row["language"],
                    synthetic=bool(row["synthetic"]),
                    source_dataset=row["source_dataset"],
                    spoken_text=row["spoken_text"],
                )

    def class_counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT label, COUNT(*) AS count FROM examples GROUP BY label"
            )
            return {row["label"]: int(row["count"]) for row in rows}
