"""Artifact summaries with task/scenario clustered uncertainty, never best-run selection."""

import random
import statistics


def describe(values):
    observed = sorted(float(v) for v in values if v is not None)

    def percentile(p):
        if not observed:
            return None
        index = (len(observed) - 1) * p
        low = int(index)
        high = min(low + 1, len(observed) - 1)
        return observed[low] + (observed[high] - observed[low]) * (index - low)

    return {
        "n": len(observed),
        "missing": len(values) - len(observed),
        "mean": statistics.mean(observed) if observed else None,
        "sample_sd": statistics.stdev(observed) if len(observed) > 1 else None,
        "p50": percentile(0.5),
        "p95": percentile(0.95),
    }


def cluster_bootstrap(rows, *, value="reward", cluster="task_id", resamples=1000, seed=42):
    groups = {}
    for row in rows:
        if row.get(value) is not None:
            groups.setdefault(row[cluster], []).append(float(row[value]))
    if len(groups) < 2:
        return {
            "low": None,
            "high": None,
            "clusters": len(groups),
            "undefined_reason": "fewer than two observed clusters",
        }
    rng = random.Random(seed)
    keys = sorted(groups)
    means = []
    for _ in range(resamples):
        sample = [v for key in rng.choices(keys, k=len(keys)) for v in groups[key]]
        means.append(statistics.mean(sample))
    means.sort()
    return {
        "low": means[int(0.025 * (resamples - 1))],
        "high": means[int(0.975 * (resamples - 1))],
        "clusters": len(groups),
        "undefined_reason": None,
        "resamples": resamples,
        "seed": seed,
    }


def paired_deltas(left, right, *, value="reward", keys=("task_id", "seed", "trial")):
    def index(rows):
        result = {}
        for row in rows:
            key = tuple(row.get(k) for k in keys)
            if key in result:
                raise ValueError("Ambiguous paired observations")
            result[key] = row
        return result

    a, b = index(left), index(right)
    rows = []
    for key in a.keys() & b.keys():
        av, bv = a[key].get(value), b[key].get(value)
        if av is not None and bv is not None:
            rows.append(dict(zip(keys, key, strict=True)) | {"delta": bv - av})
    return {
        "pairs": len(rows),
        "unmatched": len(a.keys() ^ b.keys()),
        "summary": describe([r["delta"] for r in rows]),
        "ci95": cluster_bootstrap(rows, value="delta", cluster=keys[0]),
        "rows": rows,
    }
