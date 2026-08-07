# Training Compute Backend

Phase 2.8.2H adds one governed compute layer to the existing LightGBM Ranker training path. The
Ranker baseline, production training, Challenger training, and governed retraining all reuse the
same trainer and may explicitly request `cpu` or `cuda`. Feature sets, labels, folds, seeds, and
semantic LightGBM parameters are unchanged. In particular, this phase does not change `max_bin`.

Diagnostics remain CPU-only because their deterministic assumptions have not been qualified on
CUDA. Model loading and prediction also remain CPU-compatible: CUDA training publishes the normal
LightGBM `model.txt`, and daily inference, validation prediction, Shadow, Paper Trading, monitoring,
and governance do not require a GPU.

## Configuration

The safe repository default is CPU:

```yaml
ranker:
  training_backend:
    device_type: cpu
    gpu_device_id: 0
    allow_cpu_fallback: false
    require_cuda_probe: true
```

Only `cpu` and `cuda` are accepted. CUDA execution must be selected explicitly. When CUDA is
requested, the application runs a tiny in-memory LightGBM `lambdarank` smoke fit before governed
training. This verifies the installed Python package, not merely the presence of `nvidia-smi` or a
CUDA toolkit. CUDA failure blocks training when `allow_cpu_fallback` is false. An explicitly enabled
fallback records requested CUDA, effective CPU, the probe result, and the fallback reason.

Inspect the configured backend without publishing artifacts:

```bash
ashare-quant --config config/default.yaml models training-backend-status
```

Exit code `0` means an effective backend is available, `1` means the configured backend is
unavailable, and `2` means configuration or probe failure. Governed retraining and qualification
perform the same check in addition to readiness, authorization, budget, cooldown, source-integrity,
and lock checks. Authorization cannot bypass an unavailable requested backend.

## CUDA Environment

The project does not install NVIDIA drivers, the CUDA toolkit, or compile LightGBM automatically.
On the training host, verify the environment explicitly:

```bash
nvidia-smi
nvcc --version
python -c "import lightgbm as lgb; print(lgb.__version__)"
```

If the current LightGBM package lacks CUDA support, build the operator-selected validated version:

```bash
pip uninstall -y lightgbm
pip install --no-cache-dir --no-binary lightgbm \
  --config-settings=cmake.define.USE_CUDA=ON \
  "lightgbm==<validated-version>"
```

Then set `device_type: cuda`, keep `allow_cpu_fallback: false` for governed runs, and run
`models training-backend-status`. Stop if the probe is not `AVAILABLE`.

## Provenance

New model manifests record `training_compute`, including requested and effective device, GPU index,
fallback state, LightGBM version, and probe result. Governed candidate registrations carry the same
metadata and its hash. CPU and CUDA executions have different training execution identities even
when their modeling identity is identical. Hardware descriptions and probe timestamps are audit
metadata, not logical model inputs. Legacy manifests are not rewritten and may have no
`training_compute` block.

CUDA floating-point behavior may differ from CPU behavior. Model files and predictions are not
expected to be bitwise identical. Seeds and semantic inputs remain fixed, and the benchmark below
measures behavioral consistency.

## Controlled Benchmark

Choose one immutable model experiment whose manifest freezes train and validation periods, fold,
feature list, horizon, and source lineage. Never use final-test data for backend selection.

```bash
ashare-quant --config config/default.yaml models benchmark-training-backend \
  --backend cpu --experiment-id EXPERIMENT_ID

ashare-quant --config config/default.yaml models benchmark-training-backend \
  --backend cuda --experiment-id EXPERIMENT_ID

ashare-quant --config config/default.yaml models compare-training-backends \
  --cpu-benchmark-id CPU_BENCHMARK_ID \
  --cuda-benchmark-id CUDA_BENCHMARK_ID
```

Artifacts are isolated under `reports/training_backend_benchmarks/` and published atomically with
the manifest last. Comparison requires exact source, feature, fold, period, horizon, semantic
parameter, and seed identity. It evaluates prediction Pearson/Spearman correlation, Rank IC and
NDCG deltas, feature-importance similarity, portfolio proxy deltas, wall time, and speedup.

Correctness determines `PASS` or `FAIL`; speedup is informational. These tolerances are engineering
qualification settings and are not Promotion gates. A fast CUDA result with divergent metrics
fails. Do not enable CUDA for controlled training until the comparison passes and its evidence has
been reviewed.

## Troubleshooting

- `nvidia-smi` succeeds but the probe fails: the installed LightGBM wheel may be CPU-only.
- Invalid GPU ID: correct `gpu_device_id`; do not enable fallback to hide configuration errors.
- CUDA out-of-memory: stop and review workload sizing. This phase does not change binning or model
  hyperparameters automatically.
- CPU fallback appears in a manifest: treat the artifact as CPU-executed despite the CUDA request.
- Metric consistency fails: preserve both benchmark artifacts and investigate; do not weaken the
  tolerances or promote the CUDA artifact automatically.
