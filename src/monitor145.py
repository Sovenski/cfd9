"""Monitoring helpers for Speculatores 14.5 notebook runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState


@dataclass(frozen=True)
class StudyProgress:
    side: str
    study_name: str
    completed: int
    pruned: int
    failed: int
    running: int
    total_done: int
    target: int
    best_value: float | None
    median_value: float | None


def make_storage(storage_path: str | Path) -> JournalStorage:
    storage_path = Path(storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    lock_obj = JournalFileOpenLock(str(storage_path))
    return JournalStorage(JournalFileBackend(str(storage_path), lock_obj=lock_obj))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def summarize_study(
    storage_path: str | Path,
    study_name: str,
    side: str,
    target: int,
) -> StudyProgress:
    storage = make_storage(storage_path)
    study = optuna.load_study(study_name=study_name, storage=storage)
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == TrialState.FAIL]
    running_trials = [t for t in study.trials if t.state == TrialState.RUNNING]
    values = [float(t.value) for t in complete_trials if t.value is not None]
    return StudyProgress(
        side=side,
        study_name=study_name,
        completed=len(complete_trials),
        pruned=len(pruned_trials),
        failed=len(failed_trials),
        running=len(running_trials),
        total_done=len(complete_trials) + len(pruned_trials) + len(failed_trials),
        target=target,
        best_value=(max(values) if values else None),
        median_value=_median(values),
    )


def summarize_run(
    storage_path: str | Path,
    study_prefix: str,
    dataset_version: str,
    target: int,
) -> dict[str, StudyProgress]:
    result: dict[str, StudyProgress] = {}
    for side in ("high", "low"):
        study_name = f"{study_prefix}_{dataset_version}_{side}"
        result[side] = summarize_study(storage_path, study_name, side, target)
    return result


def progress_dict(progress: StudyProgress) -> dict[str, Any]:
    return {
        "side": progress.side,
        "study_name": progress.study_name,
        "completed": progress.completed,
        "pruned": progress.pruned,
        "failed": progress.failed,
        "running": progress.running,
        "total_done": progress.total_done,
        "target": progress.target,
        "best_value": progress.best_value,
        "median_value": progress.median_value,
    }
