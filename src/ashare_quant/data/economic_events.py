"""Evidence-only economic events. Nothing here authorizes portfolio mutations."""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ashare_quant.backtest.continuous_source import payload_hash
from ashare_quant.data.exceptions import DataValidationError

CONTRACT = "economic_event_evidence_v1"
NORMALIZATION = "economic_corporate_action_catalog_v1"
TERMINAL_CONTRACT = "terminal_economic_event_v1"
DIVIDEND_FIELDS = (
    "ts_code",
    "end_date",
    "ann_date",
    "div_proc",
    "stk_div",
    "stk_bo_rate",
    "stk_co_rate",
    "cash_div",
    "cash_div_tax",
    "record_date",
    "ex_date",
    "pay_date",
    "div_listdate",
    "imp_ann_date",
    "base_date",
    "base_share",
)
DATES = {
    "ann_date": "announcement_date",
    "imp_ann_date": "implementation_announcement_date",
    "record_date": "record_date",
    "ex_date": "ex_date",
    "pay_date": "cash_pay_date",
    "div_listdate": "share_listing_date",
}
AMOUNTS = {
    "cash_div": "cash_div_after_tax_per_share",
    "cash_div_tax": "cash_div_before_tax_per_share",
    "stk_bo_rate": "stock_bonus_rate",
    "stk_co_rate": "capital_reserve_conversion_rate",
    "stk_div": "total_stock_distribution_rate",
}
EVENT_COLUMNS = [
    "event_id",
    "ts_code",
    "provider_source_id",
    "provider_raw_row_hash",
    "provider_raw_row_hashes",
    *DATES.values(),
    *AMOUNTS.values(),
    "component_type",
    "evidence_status",
    "issues",
    "normalization_contract_version",
    "cash_tax_semantics",
    "evidence_use",
    "announcement_timing",
    "within_execution_support",
]


def require(condition: bool, code: str) -> None:
    """Fail closed with an evidence-domain error."""
    if not condition:
        raise DataValidationError(f"ECONOMIC_EVIDENCE_{code}")


def date_value(value: object) -> str | None:
    """Nullable real Gregorian dates, never sentinel or inferred dates."""
    if value is None or value == "":
        return None
    text = str(value)
    try:
        valid = len(text) == 8 and datetime.strptime(text, "%Y%m%d").strftime("%Y%m%d") == text
    except ValueError:
        valid = False
    require(valid and not text.startswith("9999"), "INVALID_DATE")
    return text


def amount(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError) as exc:
        raise DataValidationError("ECONOMIC_EVIDENCE_INVALID_AMOUNT") from exc
    require(math.isfinite(result) and result >= 0, "INVALID_AMOUNT")
    return result


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DividendReview(StrictRecord):
    provider_source_id: str
    provider_raw_row_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["APPROVED", "REJECTED", "PENDING"]
    reviewed_by: str
    reviewed_fact: str

    @model_validator(mode="after")
    def reviewed(self) -> DividendReview:
        if self.decision != "PENDING":
            require(bool(self.reviewed_by.strip() and self.reviewed_fact.strip()), "REVIEW_MISSING")
        return self


class TerminalEconomicRecord(StrictRecord):
    """Human-reviewed claim tied to frozen document bytes, not a listing inference."""

    ts_code: str
    terminal_date: str
    economic_resolution_type: Literal[
        "CASH_SETTLEMENT",
        "SUCCESSOR_SHARE_ENTITLEMENT",
        "TRANSFERRED_SECURITY_RIGHT",
        "LIQUIDATION_RECEIVABLE",
        "ZERO_RECOVERY_PROVEN",
        "ECONOMIC_RECOVERY_UNKNOWN",
    ]
    cash_per_share: float | None = None
    successor_security: str | None = None
    share_conversion_ratio: float | None = None
    receivable_terms: str | None = None
    effective_date: str | None = None
    settlement_date: str | None = None
    source_type: Literal["EXCHANGE", "ISSUER_DISCLOSURE", "CSDC", "COURT", "LIQUIDATOR"]
    source_url: str
    document_id: str
    document_file: str
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_location: str
    reviewed_fact: str
    reviewed_by: str
    review_status: Literal["VERIFIED"]
    evidence_scope: Literal["ECONOMIC_ENTITLEMENT"]
    explicit_zero_recovery: bool = False

    @model_validator(mode="after")
    def semantics(self) -> TerminalEconomicRecord:
        for v in (self.terminal_date, self.effective_date, self.settlement_date):
            date_value(v)
        self.effective_date = date_value(self.effective_date)
        self.settlement_date = date_value(self.settlement_date)
        for numeric in (self.cash_per_share, self.share_conversion_ratio):
            amount(numeric)
        require(bool(self.ts_code and self.terminal_date), "TERMINAL_IDENTITY_MISSING")
        require(
            all(
                x.strip()
                for x in (
                    self.source_url,
                    self.document_id,
                    self.document_location,
                    self.reviewed_fact,
                    self.reviewed_by,
                )
            ),
            "TERMINAL_REVIEW_MISSING",
        )
        if self.economic_resolution_type == "ZERO_RECOVERY_PROVEN":
            require(
                self.explicit_zero_recovery
                and self.cash_per_share == 0
                and self.successor_security is None
                and self.receivable_terms is None
                and self.share_conversion_ratio is None,
                "ZERO_RECOVERY_NOT_PROVEN",
            )
        else:
            require(not self.explicit_zero_recovery, "ZERO_RECOVERY_CONFLICT")
        if self.share_conversion_ratio is not None:
            require(
                self.share_conversion_ratio > 0 and bool(self.successor_security),
                "SHARE_TERMS_INVALID",
            )
        if self.economic_resolution_type == "CASH_SETTLEMENT":
            require(
                self.successor_security is None
                and self.share_conversion_ratio is None
                and self.receivable_terms is None,
                "TERMINAL_COMPONENT_CONFLICT",
            )
        if self.economic_resolution_type == "SUCCESSOR_SHARE_ENTITLEMENT":
            require(
                self.cash_per_share is None and self.receivable_terms is None,
                "TERMINAL_COMPONENT_CONFLICT",
            )
        if self.economic_resolution_type in (
            "TRANSFERRED_SECURITY_RIGHT",
            "LIQUIDATION_RECEIVABLE",
        ):
            require(
                self.cash_per_share is None and self.share_conversion_ratio is None,
                "TERMINAL_COMPONENT_CONFLICT",
            )
        if self.economic_resolution_type == "ECONOMIC_RECOVERY_UNKNOWN":
            require(
                all(
                    v is None
                    for v in (
                        self.cash_per_share,
                        self.successor_security,
                        self.share_conversion_ratio,
                        self.receivable_terms,
                    )
                ),
                "UNKNOWN_HAS_TERMS",
            )
        return self

    def complete(self) -> bool:
        if self.economic_resolution_type == "ZERO_RECOVERY_PROVEN":
            return self.effective_date is not None
        if self.economic_resolution_type == "CASH_SETTLEMENT":
            return self.cash_per_share is not None and self.settlement_date is not None
        if self.economic_resolution_type == "SUCCESSOR_SHARE_ENTITLEMENT":
            return bool(
                self.successor_security and self.share_conversion_ratio and self.settlement_date
            )
        if self.economic_resolution_type in (
            "TRANSFERRED_SECURITY_RIGHT",
            "LIQUIDATION_RECEIVABLE",
        ):
            return bool(self.receivable_terms and self.effective_date)
        return False


def normalize_dividends(
    raw: pd.DataFrame,
    source_id: str,
    reviews: list[dict[str, Any]],
    cutoff: str,
) -> pd.DataFrame:
    """Deterministic provider candidates, with explicit review and conflict gates."""
    by_hash: dict[str, DividendReview] = {}
    for reviewed_item in reviews:
        validated_review = DividendReview.model_validate(reviewed_item)
        require(validated_review.provider_source_id == source_id, "REVIEW_SOURCE_MISMATCH")
        require(validated_review.provider_raw_row_hash not in by_hash, "DUPLICATE_REVIEW")
        by_hash[validated_review.provider_raw_row_hash] = validated_review
    require(set(by_hash) <= set(raw.provider_raw_row_hash), "REVIEW_ROW_MISSING")
    rows: list[dict[str, Any]] = []
    for item in raw.to_dict("records"):
        body = json.loads(item["raw_row_json"])
        # Only the documented implemented state. Proposals remain in raw evidence.
        if body.get("div_proc") != "实施":
            continue
        issues: list[str] = []
        values: dict[str, Any] = {}
        for field, name in {**DATES, **AMOUNTS}.items():
            try:
                values[name] = (
                    date_value(body.get(field)) if field in DATES else amount(body.get(field))
                )
            except DataValidationError:
                values[name] = None
                issues.append(f"INVALID_{field}")
        cash = values["cash_div_before_tax_per_share"]
        after = values["cash_div_after_tax_per_share"]
        bonus = values["stock_bonus_rate"]
        reserve = values["capital_reserve_conversion_rate"]
        total = values["total_stock_distribution_rate"]
        if any(values[name] is None for name in AMOUNTS.values()):
            issues.append("MISSING_ECONOMIC_COMPONENTS")
        if None not in (bonus, reserve, total) and not math.isclose(
            bonus + reserve, total, abs_tol=1e-10, rel_tol=1e-8
        ):
            issues.append("CONFLICTING_SHARE_COMPONENTS")
        if cash is not None and after is not None and after > cash + 1e-10:
            issues.append("CONFLICTING_TAX_COMPONENTS")
        components = []
        if (cash or 0) > 0 or (after or 0) > 0:
            components.append("CASH_DIVIDEND")
        if (bonus or 0) > 0:
            components.append("STOCK_BONUS")
        if (reserve or 0) > 0:
            components.append("CAPITAL_RESERVE_CONVERSION")
        if (total or 0) > 0 and (bonus is None or reserve is None):
            issues.append("SHARE_COMPONENTS_UNKNOWN")
        for field in ("record_date", "ex_date"):
            if values[field] is None:
                issues.append(f"MISSING_{field}")
        if "CASH_DIVIDEND" in components and values["cash_pay_date"] is None:
            issues.append("MISSING_pay_date")
        if (total or 0) > 0 and values["share_listing_date"] is None:
            issues.append("MISSING_div_listdate")
        rd, ex = values["record_date"], values["ex_date"]
        if rd and ex and rd >= ex:
            issues.append("DATE_ORDER_CONFLICT")
        for field in ("cash_pay_date", "share_listing_date"):
            if ex and values[field] and values[field] < ex:
                issues.append("DATE_ORDER_CONFLICT")
        review = by_hash.get(item["provider_raw_row_hash"])
        state = (
            "REVIEWED_PROVIDER_EVIDENCE"
            if review and review.decision == "APPROVED"
            else "UNREVIEWED"
        )
        if review and review.decision == "REJECTED":
            state = "REJECTED"
        if issues:
            state = "AMBIGUOUS_PROVIDER_EVENT"
        component = components[0] if len(components) == 1 else "COMPOSITE_DISTRIBUTION"
        if not components:
            component = "UNSUPPORTED_OTHER_EVENT"
        record = {
            "ts_code": item["ts_code"],
            "provider_source_id": source_id,
            "provider_raw_row_hash": item["provider_raw_row_hash"],
            "provider_raw_row_hashes": json.dumps([item["provider_raw_row_hash"]]),
            **values,
            "component_type": component,
            "evidence_status": state,
            "issues": json.dumps(sorted(set(issues))),
            "normalization_contract_version": NORMALIZATION,
            "cash_tax_semantics": "UNRESOLVED_EXECUTION_POLICY",
            "evidence_use": "POST_HOC_ACCOUNTING_EVIDENCE",
            "announcement_timing": "POST_EVENT"
            if ex
            and any(
                values[f] and values[f] > ex
                for f in ("announcement_date", "implementation_announcement_date")
            )
            else "NOT_PROVEN_POST_EVENT",
            "within_execution_support": bool(ex and ex <= cutoff),
        }
        record["event_id"] = "economic_event_" + payload_hash(record)[:24]
        rows.append(record)
    result = pd.DataFrame(rows, columns=EVENT_COLUMNS).drop_duplicates("event_id")
    # Neither a later revision nor a closer factor ratio may choose conflicting economics.
    economics = [*DATES.values(), *AMOUNTS.values()]
    for _, group in result[result.ex_date.notna()].groupby(["ts_code", "ex_date"]):
        if len(group[economics].drop_duplicates()) > 1:
            result.loc[group.index, "evidence_status"] = "MULTIPLE_CONFLICTING_EVENTS"
    # Equivalent alias responses retain every raw reference but one economic record.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in result.astype(object).where(result.notna(), None).to_dict("records"):
        semantic = {
            str(k): v
            for k, v in row.items()
            if k not in {"event_id", "provider_raw_row_hash", "provider_raw_row_hashes"}
        }
        grouped.setdefault(payload_hash(semantic), []).append({str(k): v for k, v in row.items()})
    deduplicated = []
    for group_rows in grouped.values():
        hashes = sorted({r["provider_raw_row_hash"] for r in group_rows})
        merged = group_rows[0].copy()
        merged["provider_raw_row_hash"] = hashes[0]
        merged["provider_raw_row_hashes"] = json.dumps(hashes)
        merged["event_id"] = (
            "economic_event_"
            + payload_hash({k: v for k, v in merged.items() if k != "event_id"})[:24]
        )
        deduplicated.append(merged)
    return (
        pd.DataFrame(deduplicated, columns=EVENT_COLUMNS)
        .sort_values("event_id")
        .reset_index(drop=True)
    )
