from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

from ashare_quant.cli import run_production_cli_command
from ashare_quant.orchestration.lock import (
    ProductionLockError,
    acquire_production_lock,
    detect_production_lock_owner,
    release_production_lock,
)


def test_acquire_and_release_production_lock(tmp_path: Path) -> None:
    path = tmp_path / "runs" / ".production.lock"

    lock = acquire_production_lock(path, command="pytest acquire")
    owner = detect_production_lock_owner(path)

    assert owner is not None
    assert owner.pid == os.getpid()
    assert owner.command == "pytest acquire"
    assert not lock.released

    release_production_lock(lock)

    assert lock.released
    assert detect_production_lock_owner(path) is None


def test_second_process_cannot_acquire_production_lock(tmp_path: Path) -> None:
    path = tmp_path / "runs" / ".production.lock"
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    lock = acquire_production_lock(path, command="owner process")
    process = context.Process(target=_try_acquire, args=(path, result_queue))

    try:
        process.start()
        status, message = result_queue.get(timeout=10)
        process.join(timeout=10)
    finally:
        release_production_lock(lock)

    assert process.exitcode == 0
    assert status == "blocked"
    assert "another production run is active" in message
    assert f"pid={os.getpid()}" in message


def test_lock_is_released_when_owner_process_fails(tmp_path: Path) -> None:
    path = tmp_path / "runs" / ".production.lock"
    context = multiprocessing.get_context("spawn")
    ready_queue = context.Queue()
    process = context.Process(target=_acquire_then_fail, args=(path, ready_queue))

    process.start()
    assert ready_queue.get(timeout=10) == "acquired"
    process.join(timeout=10)

    assert process.exitcode == 7
    replacement = acquire_production_lock(path, command="replacement process")
    release_production_lock(replacement)


def test_cli_wrapper_returns_nonzero_when_production_lock_is_held(tmp_path: Path, capsys) -> None:
    path = tmp_path / "runs" / ".production.lock"
    lock = acquire_production_lock(path, command="active pipeline")
    called = False

    def operation() -> int:
        nonlocal called
        called = True
        return 0

    try:
        exit_code = run_production_cli_command(operation, lock_path=path)
    finally:
        release_production_lock(lock)

    captured = capsys.readouterr()
    assert exit_code == 3
    assert not called
    assert "production run blocked" in captured.err
    assert "active pipeline" in captured.err


def _try_acquire(path: Path, result_queue) -> None:
    try:
        lock = acquire_production_lock(path, command="contending process")
    except ProductionLockError as error:
        result_queue.put(("blocked", str(error)))
        return
    release_production_lock(lock)
    result_queue.put(("acquired", ""))


def _acquire_then_fail(path: Path, ready_queue) -> None:
    acquire_production_lock(path, command="failing process")
    ready_queue.put("acquired")
    ready_queue.close()
    ready_queue.join_thread()
    os._exit(7)
