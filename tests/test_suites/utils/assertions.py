from typing import Any

from .criterions import AlwaysPass, Criterion
from .metric import Metric


def check(
    metric: Metric, reference: Any, experiment: Any, criterion: Criterion | None = None
) -> None:
    """Run a metric (pure stats), then judge with a post-metric Criterion."""
    stat = metric.run(reference, experiment)
    crit = criterion or AlwaysPass()
    judged = crit.judge(stat)
    if not judged.passed:
        details = ", ".join(f"{k}={v}" for k, v in judged.details.items())
        extra = f" ({details})" if details else ""
        raise AssertionError(f"[{metric.id}] failed{extra}")
