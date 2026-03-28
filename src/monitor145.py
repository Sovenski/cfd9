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
    p75_value: float | None


@dataclass(frozen=True)
class RunProgress:
    total_done: int
    target: int
    running: int
    completed: int
    pruned: int
    failed: int
    high: StudyProgress
    low: StudyProgress


def make_storage(storage_path: str | Path) -> JournalStorage:
    storage_path = Path(storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    lock_obj = JournalFileOpenLock(str(storage_path))
    return JournalStorage(JournalFileBackend(str(storage_path), lock_obj=lock_obj))


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    weight = pos - low
    return values[low] * (1.0 - weight) + values[high] * weight


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
        p75_value=_percentile(values, 0.75),
    )


def summarize_run(
    storage_path: str | Path,
    study_prefix: str,
    dataset_version: str,
    target: int,
) -> RunProgress:
    studies: dict[str, StudyProgress] = {}
    for side in ("high", "low"):
        study_name = f"{study_prefix}_{dataset_version}_{side}"
        studies[side] = summarize_study(storage_path, study_name, side, target)
    return RunProgress(
        total_done=studies["high"].total_done + studies["low"].total_done,
        target=(2 * target),
        running=studies["high"].running + studies["low"].running,
        completed=studies["high"].completed + studies["low"].completed,
        pruned=studies["high"].pruned + studies["low"].pruned,
        failed=studies["high"].failed + studies["low"].failed,
        high=studies["high"],
        low=studies["low"],
    )


def progress_dict(progress: StudyProgress | RunProgress) -> dict[str, Any]:
    if isinstance(progress, RunProgress):
        return {
            "total_done": progress.total_done,
            "target": progress.target,
            "running": progress.running,
            "completed": progress.completed,
            "pruned": progress.pruned,
            "failed": progress.failed,
        }
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
        "p75_value": progress.p75_value,
    }
