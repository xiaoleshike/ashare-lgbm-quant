# Production Run Manifest

Future production commands can create an auditable run record under
`runs/YYYYMMDD/<run_id>/manifest.json`. A new run starts as `running`; it can become `success` only
after every declared stage succeeds. An exception should call `record_failure()`. A process killed
before its exception handler runs leaves `running`, never a false success.

```python
from ashare_quant.orchestration import (
    create_run,
    record_failure,
    record_stage_end,
    record_stage_start,
    update_run_status,
)

run = create_run(
    "ashare-quant pipeline daily",
    config_path="config/default.yaml",
    stages=("data_update",),
)
try:
    record_stage_start(run, "data_update")
    update_data()
    record_stage_end(run, "data_update")
    update_run_status(run, "success")
except Exception as error:
    record_failure(run, error)
    raise
```

Manifest updates use a temporary file followed by atomic replacement, so readers observe either
the complete previous state or the complete new state. Future pipeline commands must hold the
production lock for the entire run; the manifest provides audit state and does not replace the
cross-process lock. Completed stages may also record an elapsed duration and a structured result
such as the delegated command, exit code, as-of date, and generated artifact manifest.
