from __future__ import annotations

import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class ParallelStats:
    stage: str
    total: int
    workers: int
    succeeded: int
    failed: int
    elapsed_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "total": self.total,
            "workers": self.workers,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "elapsed_seconds": self.elapsed_seconds,
        }


def resolve_workers(requested: int, total: int) -> int:
    """Resolve a user worker setting.

    Semantics:
      * 0  -> one worker per selected case (all cases submitted concurrently)
      * -1 -> a conservative automatic thread count
      * >0 -> exactly that many workers, capped by the task count

    The executor never creates fewer than one worker when work exists.
    """
    if total <= 0:
        return 0
    requested = int(requested)
    if requested == 0:
        return total
    if requested < 0:
        automatic = min(32, (os.cpu_count() or 1) + 4)
        return max(1, min(total, automatic))
    return max(1, min(total, requested))


def _public_item(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    return {key: value for key, value in item.items() if not str(key).startswith("_")}


def _item_label(item: Any, index: int) -> str:
    if isinstance(item, (str, int, float)):
        return str(item)
    if isinstance(item, dict):
        for key in ("sample_id", "question_id", "name", "id"):
            value = item.get(key)
            if value:
                return str(value)
    return str(index)



def should_checkpoint(completed: int, total: int, every: int = 10) -> bool:
    every = max(1, int(every))
    return completed >= total or completed % every == 0

def run_parallel(
    items: Sequence[T] | Iterable[T],
    worker: Callable[[T], R],
    *,
    workers: int = 0,
    stage: str = "stage",
    merge_input: bool = False,
    fail_fast: bool = False,
    progress: bool = True,
    on_result: Optional[Callable[[int, R, int, int], None]] = None,
) -> tuple[List[R], ParallelStats]:
    """Run independent case tasks concurrently and preserve manifest order.

    Exceptions are converted into a deterministic failure dictionary unless
    ``fail_fast`` is enabled. ``on_result`` is executed in the coordinator
    thread, which makes it safe to write incremental manifests/checkpoints.
    """
    values = list(items)
    total = len(values)
    effective_workers = resolve_workers(workers, total)
    started = time.time()
    if total == 0:
        stats = ParallelStats(stage, 0, 0, 0, 0, 0.0)
        return [], stats

    ordered: List[Optional[R]] = [None] * total
    completed = 0
    failed = 0

    def call(index: int, item: T) -> R:
        try:
            result = worker(item)
            if merge_input and isinstance(result, dict):
                return {**_public_item(item), **result}  # type: ignore[return-value]
            return result
        except Exception as exc:
            if fail_fast:
                raise
            base = _public_item(item) if merge_input else {}
            return {**base, "status": "failed", "error": repr(exc)}  # type: ignore[return-value]

    def accept(index: int, result: R) -> None:
        nonlocal completed, failed
        ordered[index] = result
        completed += 1
        if isinstance(result, dict) and result.get("status") == "failed":
            failed += 1
        if progress:
            status = result.get("status", "ok") if isinstance(result, dict) else "ok"
            label = _item_label(values[index], index)
            elapsed = time.time() - started
            print(
                f"[{stage}] {completed}/{total} | {label} | {status} | "
                f"workers={effective_workers} | elapsed={elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        if on_result is not None:
            on_result(index, result, completed, total)

    if effective_workers <= 1:
        for index, item in enumerate(values):
            result = call(index, item)
            accept(index, result)
    else:
        with ThreadPoolExecutor(
            max_workers=effective_workers,
            thread_name_prefix=f"dimmem-{stage}",
        ) as executor:
            futures: Dict[Future[R], int] = {
                executor.submit(call, index, item): index
                for index, item in enumerate(values)
            }
            try:
                for future in as_completed(futures):
                    index = futures[future]
                    result = future.result()
                    accept(index, result)
            except Exception:
                for future in futures:
                    future.cancel()
                raise

    results: List[R] = [value for value in ordered if value is not None]
    elapsed = time.time() - started
    stats = ParallelStats(
        stage=stage,
        total=total,
        workers=effective_workers,
        succeeded=total - failed,
        failed=failed,
        elapsed_seconds=elapsed,
    )
    return results, stats
