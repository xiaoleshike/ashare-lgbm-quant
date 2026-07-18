# Production Run Lock

Future production pipeline commands must use the repository-level Linux `flock` at
`runs/.production.lock`. The lock is non-blocking: a competing run fails immediately with the
active owner's PID, host, acquisition time, and command. The lock file remains on disk, but its
kernel lock is automatically released when the owning descriptor closes or the process exits.

CLI handlers should use the reusable wrapper:

```python
from ashare_quant.cli import run_production_cli_command


def run_daily_command() -> int:
    return run_production_cli_command(
        lambda: run_daily_pipeline(),
        command="ashare-quant pipeline daily",
    )
```

Lower-level orchestration code may use `production_lock()` as a context manager or call
`acquire_production_lock()` and `release_production_lock()` directly. Do not delete the lock file
while another process may be running: unlinking it can create a second inode and permit two jobs
to hold different locks at the same path.
