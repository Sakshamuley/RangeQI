"""Suitability scoring for EV commute matching."""

from __future__ import annotations

from typing import Literal

SuitabilityVerdict = Literal["suitable", "marginal", "not_suitable"]
SafeVerdict = Literal["safe", "risky"]


def compute_suitability(
    predicted_median_range_km: float,
    predicted_p10_range_km: float,
    daily_requirement_km: float,
    claimed_range_km: float,
) -> dict:
    """
    Suitability based on buffer = (predicted - requirement) / requirement.
    - Suitable: buffer > 30%
    - Marginal: 5–30%
    - Not suitable: < 5%

    Safe flag uses worst-case (p10) range for conservative commuters.
    """
    if daily_requirement_km <= 0:
        daily_requirement_km = 1.0

    buffer = (predicted_median_range_km - daily_requirement_km) / daily_requirement_km
    buffer_pct = round(buffer * 100, 1)

    if buffer > 0.30:
        verdict: SuitabilityVerdict = "suitable"
    elif buffer >= 0.05:
        verdict = "marginal"
    else:
        verdict = "not_suitable"

    # Conservative safe check on 10th percentile
    p10_buffer = (predicted_p10_range_km - daily_requirement_km) / daily_requirement_km
    safe_verdict: SafeVerdict = "safe" if p10_buffer > 0.15 else "risky"

    claimed_buffer = (claimed_range_km - daily_requirement_km) / daily_requirement_km

    return {
        "verdict": verdict,
        "safe_verdict": safe_verdict,
        "buffer_pct": buffer_pct,
        "p10_buffer_pct": round(p10_buffer * 100, 1),
        "daily_requirement_km": daily_requirement_km,
        "claimed_vs_requirement_pct": round(claimed_buffer * 100, 1),
        "message": _verdict_message(verdict, safe_verdict, buffer_pct),
    }


def _verdict_message(verdict: str, safe: str, buffer_pct: float) -> str:
    if verdict == "suitable" and safe == "safe":
        return f"Strong match — {buffer_pct:.0f}% range buffer even at median estimate."
    if verdict == "suitable":
        return f"Median range looks good ({buffer_pct:.0f}% buffer), but worst-case days may be tight."
    if verdict == "marginal":
        return f"Borderline — only {buffer_pct:.0f}% buffer. Consider charging access or a larger battery."
    return "Daily commute exceeds realistic range on most days. Not recommended without mid-day charging."
