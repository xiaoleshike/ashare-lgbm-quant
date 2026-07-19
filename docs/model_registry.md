# Production Model Registry

The file-backed registry at `models/registry.json` tracks immutable model artifact directories. Registration records artifact identity and creates a `candidate`; it never edits or replaces `model.txt`, `feature_list.json`, `metrics.json`, or `manifest.json`.

Lifecycle transitions are explicit. Promotion validates all required files, verifies the ordered feature-list hash against the registered and declared hashes, and requires non-empty test metrics. Promoting a challenger demotes the existing champion of the same model type to `candidate`. Retirement preserves both the registry record and artifact path. Only one `champion` may exist per model type.

Mutating operations use `models/.registry.lock`, update the registry atomically, and append an audit record under `models/registry_history/`. Registration is currently a Python service API because training remains separate from lifecycle approval:

```python
from pathlib import Path

from ashare_quant.models import ModelRegistry

registry = ModelRegistry(Path("models"))
registry.register_model(Path("models/experiment_b_robust_20260714T002645940015Z"))
```

Inspect and manage registered models with:

```bash
ashare-quant --config config/default.yaml models list
ashare-quant --config config/default.yaml models champion
ashare-quant --config config/default.yaml models promote MODEL_ID
ashare-quant --config config/default.yaml models retire MODEL_ID
```

The registry does not retrain models, infer scores, or automatically promote candidates. A missing champion is valid until an operator explicitly approves one.
