"""Cross-process production run locking backed by Linux ``flock``."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import socket
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

DEFAULT_PRODUCTION_LOCK_PATH = Path("runs/.production.lock")


@dataclass(frozen=True, slots=True)
class ProductionLockOwner:
    """Metadata describing the process that acquired the production lock."""

    pid: int | None
    hostname: str | None
    acquired_at: str | None
    command: str | None

    def describe(self) -> str:
        """Return compact owner details for an operator-facing error."""

        details = [
            f"pid={self.pid if self.pid is not None else 'unknown'}",
            f"host={self.hostname or 'unknown'}",
            f"acquired_at={self.acquired_at or 'unknown'}",
        ]
        if self.command:
            details.append(f"command={self.command}")
        return " ".join(details)


class ProductionLockError(RuntimeError):
    """Raised when another process already holds the production lock."""


@dataclass(slots=True)
class ProductionLock:
    """An acquired production lock whose open descriptor owns the kernel lock."""

    path: Path
    owner: ProductionLockOwner
    _file: IO[str]
    _released: bool = False

    @property
    def released(self) -> bool:
        """Return whether this handle has already released its lock."""

        return self._released


def acquire_production_lock(
    lock_path: Path = DEFAULT_PRODUCTION_LOCK_PATH,
    *,
    command: str | None = None,
) -> ProductionLock:
    """Acquire the repository production lock without waiting.

    The returned handle must remain alive while protected work runs. The kernel
    releases the lock automatically when its descriptor closes or the process exits.
    """

    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno not in {errno.EACCES, errno.EAGAIN}:
            lock_file.close()
            raise
        owner = _read_owner(lock_file)
        lock_file.close()
        raise ProductionLockError(
            f"another production run is active ({owner.describe()}); lock={path}"
        ) from error

    owner = ProductionLockOwner(
        pid=os.getpid(),
        hostname=socket.gethostname(),
        acquired_at=datetime.now(UTC).isoformat(timespec="seconds"),
        command=command or " ".join(sys.argv),
    )
    try:
        _write_owner(lock_file, owner)
    except Exception:
        lock_file.close()
        raise
    return ProductionLock(path=path, owner=owner, _file=lock_file)


def release_production_lock(lock: ProductionLock) -> None:
    """Release an acquired production lock; repeated calls are harmless."""

    if lock._released:
        return
    try:
        lock._file.seek(0)
        lock._file.truncate()
        lock._file.flush()
    finally:
        try:
            fcntl.flock(lock._file.fileno(), fcntl.LOCK_UN)
        finally:
            lock._file.close()
            lock._released = True


def detect_production_lock_owner(
    lock_path: Path = DEFAULT_PRODUCTION_LOCK_PATH,
) -> ProductionLockOwner | None:
    """Return the active lock owner, or ``None`` when the lock is available."""

    path = Path(lock_path)
    if not path.exists():
        return None
    lock_file = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
            return _read_owner(lock_file)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return None
    finally:
        lock_file.close()


@contextmanager
def production_lock(
    lock_path: Path = DEFAULT_PRODUCTION_LOCK_PATH,
    *,
    command: str | None = None,
) -> Iterator[ProductionLock]:
    """Hold the production lock for one operation and always release it."""

    lock = acquire_production_lock(lock_path, command=command)
    try:
        yield lock
    finally:
        release_production_lock(lock)


def run_with_production_lock[T](
    operation: Callable[[], T],
    *,
    lock_path: Path = DEFAULT_PRODUCTION_LOCK_PATH,
    command: str | None = None,
) -> T:
    """Run a future pipeline or CLI operation under the production lock."""

    with production_lock(lock_path, command=command):
        return operation()


def _write_owner(lock_file: IO[str], owner: ProductionLockOwner) -> None:
    lock_file.seek(0)
    lock_file.truncate()
    json.dump(asdict(owner), lock_file, ensure_ascii=True, sort_keys=True)
    lock_file.write("\n")
    lock_file.flush()
    os.fsync(lock_file.fileno())


def _read_owner(lock_file: IO[str]) -> ProductionLockOwner:
    lock_file.seek(0)
    try:
        payload = json.load(lock_file)
    except (json.JSONDecodeError, OSError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    pid_value = payload.get("pid")
    return ProductionLockOwner(
        pid=pid_value if isinstance(pid_value, int) else None,
        hostname=_optional_string(payload.get("hostname")),
        acquired_at=_optional_string(payload.get("acquired_at")),
        command=_optional_string(payload.get("command")),
    )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
