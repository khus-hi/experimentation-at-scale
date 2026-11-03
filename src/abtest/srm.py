"""Sample ratio mismatch (SRM): did traffic split the way the design said it would?"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = ["SRM_THRESHOLD", "SRMResult", "srm_check"]

SRM_THRESHOLD = 0.001


@dataclass(frozen=True)
class SRMResult:
    """Chi-square goodness-of-fit test of observed vs expected impressions per arm."""

    observed: tuple[int, ...]
    expected: tuple[float, ...]
    chi2: float
    df: int
    p_value: float
    threshold: float
    srm: bool


def srm_check(
    impressions: Sequence[int],
    expected_ratio: Sequence[float] | None = None,
    threshold: float = SRM_THRESHOLD,
) -> SRMResult:
    r"""Chi-square goodness-of-fit test of the impression split.

    With :math:`N = \sum_i O_i` and design shares :math:`r_i` (normalised to sum to 1,
    equal by default),

    .. math::
        E_i = N r_i, \qquad \chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}, \qquad df = k - 1,

    and ``srm`` is true when :math:`p < 0.001`. Use it as the first check on any test:
    a mismatch means assignment or logging was broken, so the test's effect estimate can't
    be trusted however significant it looks. The strict threshold is the usual one, because
    with tens of thousands of tests a 0.05 cut would flag hundreds by chance.
    """
    obs = np.asarray(impressions, dtype=float)
    if obs.ndim != 1 or obs.size < 2:
        raise ValueError("need impressions for at least 2 arms")
    if (obs < 0).any():
        raise ValueError("impressions must be non-negative")
    total = obs.sum()
    if total <= 0:
        raise ValueError("total impressions must be positive")
    if expected_ratio is None:
        ratio = np.full(obs.size, 1 / obs.size)
    else:
        ratio = np.asarray(expected_ratio, dtype=float)
        if ratio.shape != obs.shape:
            raise ValueError("expected_ratio must have one entry per arm")
        if (ratio <= 0).any():
            raise ValueError("expected_ratio entries must be positive")
        ratio = ratio / ratio.sum()
    expected = total * ratio
    chi2, p = stats.chisquare(obs, expected)
    p = float(p)
    return SRMResult(
        tuple(int(o) for o in obs),
        tuple(float(e) for e in expected),
        float(chi2),
        obs.size - 1,
        p,
        threshold,
        p < threshold,
    )
