#!/usr/bin/env python3
"""Render production research artifacts as one self-contained HTML report."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from collections.abc import Callable
from html import escape
from pathlib import Path
from typing import cast

type JsonObject = dict[str, object]

RISK_LABELS = {
    "abnormal_recent_return": "当日涨跌幅异常",
    "high_volatility": "近期波动率较高",
    "low_liquidity": "成交额偏低",
    "limit_up": "当日收盘涨停",
    "missing_daily_data": "缺少当日行情",
    "missing_universe_metadata": "缺少股票池元数据",
    "missing_market_cap": "缺少市值数据",
}


def build_parser() -> argparse.ArgumentParser:
    """Create the standalone report-rendering CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True, help="Report date in YYYYMMDD.")
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path("reports"),
        help="Root containing reports/YYYYMMDD (default: reports).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: reports/YYYYMMDD/daily_report.html).",
    )
    return parser


def main() -> int:
    """Load one daily report directory and render a browser-readable HTML file."""

    args = build_parser().parse_args()
    report_dir = args.reports_root / args.as_of
    output = args.output or report_dir / "daily_report.html"
    try:
        summary = load_json_object(report_dir / "research_summary.json")
        candidates = load_candidates(report_dir / "candidates.csv")
        markdown = load_text(report_dir / "daily_report.md")
        validate_identity(summary, candidates, args.as_of)
        document = render_html(summary, candidates, markdown)
        atomic_write_text(output, document)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"daily report rendering failed: {error}")
        return 2
    print(f"daily report rendered: {output}")
    return 0


def load_json_object(path: Path) -> JsonObject:
    """Load a required JSON object."""

    if not path.is_file():
        raise ValueError(f"required research summary is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"research summary must contain a JSON object: {path}")
    return cast(JsonObject, payload)


def load_candidates(path: Path) -> list[dict[str, str]]:
    """Load candidate rows without changing their stored rank order."""

    if not path.is_file():
        raise ValueError(f"required candidate file is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"rank", "ts_code", "prediction_score", "trade_date", "model_id"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"candidate file is missing columns: {sorted(required - rows[0].keys())}")
    return sorted(rows, key=lambda row: (int(row["rank"]), row["ts_code"]))


def load_text(path: Path) -> str:
    """Load the required source Markdown report."""

    if not path.is_file():
        raise ValueError(f"required Markdown report is missing: {path}")
    return path.read_text(encoding="utf-8")


def validate_identity(summary: JsonObject, candidates: list[dict[str, str]], as_of: str) -> None:
    """Reject mixed report dates or model identities before rendering."""

    if str(summary.get("as_of")) != as_of:
        raise ValueError("research summary date does not match --as-of")
    model_id = str(summary.get("model_id", ""))
    for row in candidates:
        if row["trade_date"] != as_of:
            raise ValueError(f"candidate row date does not match --as-of: {row['ts_code']}")
        if row["model_id"] != model_id:
            raise ValueError(f"candidate model differs from research summary: {row['ts_code']}")


def render_html(
    summary: JsonObject,
    candidates: list[dict[str, str]],
    markdown: str,
) -> str:
    """Build a self-contained HTML dashboard from structured report artifacts."""

    statistics = as_object(summary.get("statistics"))
    risks = as_object_list(summary.get("risk_flags"))
    warnings = [str(item) for item in as_list(summary.get("warnings"))]
    risk_counts = Counter(str(flag) for risk in risks for flag in as_list(risk.get("flags")))
    as_of = escape(str(summary.get("as_of", "")))
    model_id = escape(str(summary.get("model_id", "")))
    candidate_count = int(summary.get("candidate_count", len(candidates)))
    prediction_count = int(summary.get("prediction_count", 0))
    top_count = int(summary.get("top_candidate_count", min(20, len(candidates))))
    board_bars = render_bars(as_object(statistics.get("board_distribution")))
    market_cap_bars = render_bars(
        as_object(statistics.get("market_cap_distribution")), translate_market_cap
    )
    industry_bars = render_bars(as_object(statistics.get("industry_distribution")))

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>A股量化研究日报 {as_of}</title>
<style>
:root {{ color-scheme: light; --ink:#17202a; --muted:#637083; --line:#d9dee5;
  --paper:#ffffff; --canvas:#f4f6f8; --accent:#176b5b; --warn:#a65b16; --risk:#a33a3a; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--canvas); color:var(--ink); font:14px/1.55 system-ui,
  -apple-system,"Segoe UI","Microsoft YaHei",sans-serif; letter-spacing:0; }}
main {{ width:min(1180px,calc(100% - 32px)); margin:24px auto 56px; }}
header {{ border-top:5px solid var(--accent); background:var(--paper); padding:22px 24px;
  border-bottom:1px solid var(--line); }}
h1 {{ margin:0 0 4px; font-size:25px; }} h2 {{ margin:0 0 14px; font-size:18px; }}
h3 {{ margin:18px 0 8px; font-size:14px; }} p {{ margin:6px 0; }}
.muted {{ color:var(--muted); }} .model {{ overflow-wrap:anywhere; }}
.kpis {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:14px 0; }}
.kpi {{ background:var(--paper); border:1px solid var(--line); border-radius:6px; padding:14px; }}
.kpi strong {{ display:block; font-size:23px; }} .kpi span {{ color:var(--muted); }}
.layout {{ display:grid; grid-template-columns:minmax(0,1.55fr) minmax(280px,.75fr);
  gap:14px; align-items:start; }}
section {{ background:var(--paper); border:1px solid var(--line); padding:18px;
  margin-bottom:14px; }}
.table-wrap {{ overflow:auto; }}
table {{ width:100%; border-collapse:collapse; white-space:nowrap; }}
th,td {{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; }}
th {{ color:var(--muted); font-weight:600; background:#f8f9fa; }}
td.num,th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.score {{ color:var(--accent); font-weight:650; }}
.risk {{ color:var(--risk); font-weight:600; }} .warning {{ color:var(--warn); }}
.bar-row {{ display:grid; grid-template-columns:minmax(110px,1fr) 2fr 38px; gap:8px;
  align-items:center; margin:7px 0; }}
.track {{ height:9px; background:#e7eaee; }} .fill {{ height:100%; background:var(--accent); }}
.tag {{ display:inline-block; border:1px solid #e5bcbc; color:var(--risk); padding:2px 6px;
  margin:1px 3px 1px 0; border-radius:4px; font-size:12px; }}
details {{ background:var(--paper); border:1px solid var(--line); padding:14px 18px; }}
summary {{ cursor:pointer; font-weight:650; }} pre {{ white-space:pre-wrap; overflow-wrap:anywhere;
  background:#f8f9fa; border:1px solid var(--line); padding:14px; }}
.notice {{ border-left:4px solid var(--warn); padding:10px 12px; background:#fff8ef; }}
@media (max-width:800px) {{ .kpis {{ grid-template-columns:repeat(2,1fr); }}
  .layout {{ grid-template-columns:1fr; }}
  main {{ width:min(100% - 18px,1180px); margin-top:10px; }} }}
@media print {{ body {{ background:#fff; }} main {{ width:100%; margin:0; }}
  details {{ display:none; }} }}
</style>
</head>
<body><main>
<header>
  <h1>A股量化研究日报</h1>
  <p class="muted">报告日期 {as_of}</p>
  <p class="model">模型：<code>{model_id}</code></p>
</header>
<div class="kpis">
  {kpi("模型打分股票", prediction_count)}
  {kpi("过滤后候选", candidate_count)}
  {kpi("报告展示", top_count)}
  {kpi("风险提示股票", len(risks))}
</div>
{render_warning_block(warnings)}
<div class="layout">
<div>
  <section><h2>候选排名 Top {top_count}</h2>
    <p class="muted">保留模型分数与候选层排名，不构成买卖建议。</p>
    {render_candidates(candidates[:top_count])}
  </section>
  <section><h2>风险提示</h2>{render_risks(risks)}</section>
</div>
<aside>
  <section><h2>板块分布</h2>{board_bars}</section>
  <section><h2>市值分布</h2>{market_cap_bars}</section>
  <section><h2>行业分布</h2>{industry_bars}</section>
  <section><h2>风险计数</h2>{render_risk_counts(risk_counts)}</section>
</aside>
</div>
<section>
  <h2>如何理解</h2>
  <p><strong>Score</strong> 仅用于同日横截面排序，数值本身不是预期收益率。</p>
  <p><strong>abnormal_recent_return</strong> 表示当日绝对涨跌幅达到配置阈值；
    <strong>high_volatility</strong> 表示近期收益波动率达到阈值。</p>
  <p>风险标签只用于研究提示，不改变模型分数、候选排序或未来交易执行。</p>
</section>
<details><summary>查看原始 Markdown 报告</summary><pre>{escape(markdown)}</pre></details>
</main></body></html>
"""


def kpi(label: str, value: int) -> str:
    """Render one compact report metric."""

    return f'<div class="kpi"><strong>{value:,}</strong><span>{escape(label)}</span></div>'


def render_candidates(rows: list[dict[str, str]]) -> str:
    """Render candidate rank and score without recomputation."""

    body = "".join(
        "<tr>"
        f'<td class="num">{int(row["rank"])}</td>'
        f"<td><code>{escape(row['ts_code'])}</code></td>"
        f'<td class="num score">{float(row["prediction_score"]):.8f}</td>'
        "</tr>"
        for row in rows
    )
    if not body:
        body = '<tr><td colspan="3" class="muted">当日没有候选股票</td></tr>'
    return (
        '<div class="table-wrap"><table><thead><tr><th class="num">排名</th>'
        '<th>股票代码</th><th class="num">模型分数</th></tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def render_risks(risks: list[JsonObject]) -> str:
    """Render structured risk records with readable labels and units."""

    if not risks:
        return '<p class="muted">未检测到配置范围内的风险提示。</p>'
    body = "".join(
        "<tr>"
        f'<td class="num">{int(risk.get("rank", 0))}</td>'
        f"<td><code>{escape(str(risk.get('ts_code', '')))}</code></td>"
        f"<td>{render_risk_tags(as_list(risk.get('flags')))}</td>"
        f'<td class="num">{format_number(risk.get("pct_chg"), "%")}</td>'
        f'<td class="num">{format_number(risk.get("recent_volatility_pct"), "%")}</td>'
        f'<td class="num">{format_amount(risk.get("amount"))}</td>'
        "</tr>"
        for risk in risks
    )
    return (
        '<div class="table-wrap"><table><thead><tr><th class="num">排名</th><th>股票代码</th>'
        '<th>风险标签</th><th class="num">当日涨跌</th><th class="num">近期波动</th>'
        f'<th class="num">成交额</th></tr></thead><tbody>{body}</tbody></table></div>'
    )


def render_bars(
    values: JsonObject,
    label_transform: Callable[[object], str] | None = None,
) -> str:
    """Render a compact CSS-only distribution chart."""

    if not values:
        return '<p class="muted">无可用数据</p>'
    numeric = {str(key): int(value) for key, value in values.items()}
    maximum = max(numeric.values(), default=1) or 1
    transform = label_transform or (lambda value: str(value))
    return "".join(
        '<div class="bar-row">'
        f"<span>{escape(str(transform(key)))}</span>"
        '<span class="track"><span class="fill" '
        f'style="width:{value / maximum * 100:.2f}%"></span></span>'
        f'<strong class="num">{value}</strong></div>'
        for key, value in numeric.items()
    )


def render_risk_tags(flags: list[object]) -> str:
    """Translate and render machine-readable risk names."""

    return "".join(
        f'<span class="tag" title="{escape(str(flag))}">'
        f"{escape(RISK_LABELS.get(str(flag), str(flag)))}</span>"
        for flag in flags
    )


def render_risk_counts(counts: Counter[str]) -> str:
    """Render counts by risk type."""

    if not counts:
        return '<p class="muted">无风险提示</p>'
    return "".join(
        f'<p><span class="risk">{escape(RISK_LABELS.get(key, key))}</span>: {value}</p>'
        for key, value in sorted(counts.items())
    )


def render_warning_block(warnings: list[str]) -> str:
    """Render source-data warnings when the report contains any."""

    if not warnings:
        return ""
    items = "".join(f"<li>{escape(item)}</li>" for item in warnings)
    return f'<section class="notice"><strong>数据警告</strong><ul>{items}</ul></section>'


def translate_market_cap(label: object) -> str:
    """Translate stable machine-readable market-cap bucket names."""

    translations = {
        "below_5bn_cny": "低于 50 亿元",
        "5bn_to_10bn_cny": "50–100 亿元",
        "10bn_to_30bn_cny": "100–300 亿元",
        "at_least_30bn_cny": "至少 300 亿元",
        "missing": "缺失",
    }
    return translations.get(str(label), str(label))


def format_number(value: object, suffix: str = "") -> str:
    """Format an optional numeric report field."""

    if not isinstance(value, (int, float)):
        return "-"
    return f"{float(value):.2f}{suffix}"


def format_amount(value: object) -> str:
    """Format Tushare amount (CNY 1,000) as CNY 100 million."""

    if not isinstance(value, (int, float)):
        return "-"
    return f"{float(value) / 100_000:.2f} 亿"


def as_object(value: object) -> JsonObject:
    """Narrow a decoded JSON value to an object."""

    return cast(JsonObject, value) if isinstance(value, dict) else {}


def as_object_list(value: object) -> list[JsonObject]:
    """Narrow a decoded JSON value to a list of objects."""

    if not isinstance(value, list):
        return []
    return [cast(JsonObject, item) for item in value if isinstance(item, dict)]


def as_list(value: object) -> list[object]:
    """Narrow a decoded JSON value to a list."""

    return cast(list[object], value) if isinstance(value, list) else []


def atomic_write_text(path: Path, content: str) -> None:
    """Publish HTML atomically without replacing a prior file on write failure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    try:
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
