# A-share LightGBM Quant

Production-oriented research scaffold for China A-share stock selection, feature research,
walk-forward validation, and backtesting. Phase 0 contains the package structure, typed
configuration, structured logging, CLI hooks, tooling, and smoke tests.

## Quick start

```bash
source .venv/bin/activate
make install
make test
make lint
make typecheck
```

Set `TUSHARE_TOKEN` in the process environment before running data-ingestion workflows.
Phase 0 does not download market data or train models.
