from casepilot.models import MetricRecord


def ratio(numerator, denominator, *, undefined_reason="zero denominator"):
    return MetricRecord(
        value=numerator / denominator if denominator else None,
        unit="ratio",
        numerator=float(numerator),
        denominator=float(denominator),
        undefined_reason=None if denominator else undefined_reason,
    )


def summarize_runs(records, *, scheduled_count=None):
    scheduled = scheduled_count if scheduled_count is not None else len(records)
    evaluated = [r for r in records if r.get("reward") is not None]
    successful = sum(r["reward"] == 1.0 for r in evaluated)
    return {
        "scheduled": scheduled,
        "evaluated": len(evaluated),
        "missing_rewards": scheduled - len(evaluated),
        "infrastructure_failures": sum(r["status"] == "infrastructure_error" for r in records),
        "task_success": ratio(successful, len(evaluated), undefined_reason="no evaluated runs"),
        "evaluation_coverage": ratio(len(evaluated), scheduled),
        "confirmed_success_lower_bound": ratio(successful, scheduled),
    }


def cost_per_success(costs, successful_count):
    if not costs or any(value is None for value in costs):
        return MetricRecord(value=None, unit="USD/success", undefined_reason="missing cohort cost")
    if not successful_count:
        return MetricRecord(value=None, unit="USD/success", undefined_reason="no successful tasks")
    return MetricRecord(value=sum(costs) / successful_count, unit="USD/success")
