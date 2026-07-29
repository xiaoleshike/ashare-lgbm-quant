"""Alpha-decay ratios for observation statistics."""

from __future__ import annotations

import math


def safe_decay_ratio(recent: float | None, historical: float | None) -> float | None:
    """Return recent divided by historical when the denominator is meaningful."""

    if recent is None or historical is None:
        return None
    if not math.isfinite(recent) or not math.isfinite(historical) or historical == 0:
        return None
    return recent / historical
