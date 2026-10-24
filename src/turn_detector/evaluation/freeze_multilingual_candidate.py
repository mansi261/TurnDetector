"""Freeze multilingual candidate and scorer identities before holdout access."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from turn_detector.training.train import read_config
from turn_detector.utils import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/multilingual_whisper_tiny_v1.json")
    )
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--serving-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    development = json.loads(args.development_report.read_text(encoding="utf-8"))
    metadata = json.loads(args.serving_metadata.read_text(encoding="utf-8"))
    candidate_hash = file_sha256(args.candidate_checkpoint)
    if development["checkpoints"]["multilingual_v1"]["sha256"] != candidate_hash:
        raise RuntimeError("development report used a different candidate")
    if metadata["checkpoint_sha256"] != candidate_hash:
        raise RuntimeError("serving export used a different candidate")
    scorer = Path(__file__).with_name("evaluate_multilingual.py")
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_status": (
            "development_quality_acceptable"
            if development["comparison"]["development_quality_acceptable"]
            else "experimental_development_quality_not_acceptable"
        ),
        "sha256": {
            "config": file_sha256(args.config),
            "baseline_checkpoint": file_sha256(args.baseline_checkpoint),
            "candidate_checkpoint": candidate_hash,
            "scoring_code": file_sha256(scorer),
        },
        "supporting_artifacts": {
            "development_report": args.development_report.resolve().as_posix(),
            "development_report_sha256": file_sha256(args.development_report),
            "serving_metadata": args.serving_metadata.resolve().as_posix(),
            "serving_metadata_sha256": file_sha256(args.serving_metadata),
        },
        "protocol": {
            "threshold": 0.5,
            "first_score_offset_seconds": config["evaluation"]["first_score_offset_seconds"],
            "replay_step_seconds": config["evaluation"]["replay_step_seconds"],
            "dispatch": "first threshold crossing per annotated pause, latched once",
        },
        "holdout_policy": "one scoring run; results cannot be used to tune this candidate",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
