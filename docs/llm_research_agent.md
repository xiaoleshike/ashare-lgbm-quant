# LLM Research Agent

The Phase 2.7.4 research agent is a read-only summarizer over immutable daily
reports. It is not part of model selection, candidate ranking, portfolio
construction, or execution.

## Commands

```bash
ashare-quant --config config/default.yaml research-agent generate --as-of YYYYMMDD
ashare-quant --config config/default.yaml research-agent validate --as-of YYYYMMDD
ashare-quant --config config/default.yaml research-agent status --as-of YYYYMMDD
```

Output is published atomically under:

```text
reports/research_agent/YYYYMMDD/
  daily_research.md
  research_summary.json
  manifest.json
```

## Provider Keys

Keys are read only from the selected provider environment variable:

| Provider | Environment variable |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Claude | `ANTHROPIC_API_KEY` |
| Gemini | `GEMINI_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |

Keys are never included in prompts, reports, manifests, or logs. If the key is
missing, the service publishes a deterministic fact-based fallback report.

## Isolation

The collector uses exact report paths and does not search the repository. It
does not access raw data, features, labels, predictions, model artifacts,
backtests, registries, or paper-trading ledgers. Source Markdown is hashed for
lineage but its text is excluded from the LLM context.

Every conclusion must cite deterministic fact IDs. Generated stock codes,
changed rankings, and unsupported numerical metrics are rejected before
publication. Advisory language is enabled by default for human review, but it
cannot change rankings, models, candidates, portfolios, or execute orders. Set
`research.agent.allow_advisory_language: false` to restore the non-advisory
policy. An invalid provider response produces the deterministic fallback.

Advice quality can be iterated through versioned prompt/config changes and
compared across immutable daily reports. The agent does not self-train and its
output is never fed automatically into model selection or trading execution.
