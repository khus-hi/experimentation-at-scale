r"""Peeking and sequential testing for a two-arm click-through test, by simulation.

What this module simulates
--------------------------
A two-arm test (control rate ``p_c``, treatment rate ``p_t = p_c (1 + lift)``) receives
``n_total`` impressions split evenly between the arms and delivered in ``n_looks`` equal daily
batches. After each batch the analyst looks at the cumulative counts and applies a decision
rule; the test stops at the first look where the rule rejects, or runs to ``n_total``.

Three rules are implemented, plus the fixed-n reference:

* ``naive``: stop the first time the pooled two-proportion z-test gives p < alpha. Each look
  is an extra chance of a false positive, so the error rate grows with the number of looks.
* ``obf``: a group-sequential design with Lan-DeMets O'Brien-Fleming-type alpha spending,
  :math:`\alpha(t) = 4 - 4\Phi(z_{1-\alpha/4} / \sqrt{t})` (alpha/2 per side). The
  boundaries are computed so the total two-sided type I error is exactly alpha over the
  planned looks.
* ``msprt``: the mixture sequential probability ratio test of Johari, Koomen, Pekelis and
  Walsh (2017), "Peeking at A/B tests", with a normal mixing distribution on the difference
  in rates. Its p-value is always valid: P(p_n < alpha for some n) <= alpha under the null,
  however often and whenever one looks.
* ``fixed``: one look at ``n_total`` (what the other rules are compared against).

Calibrated simulation, not a replay
-----------------------------------
The Upworthy Research Archive reports one row per arm with **final** impressions and clicks.
It has **no daily counts** and no test end date, so the path a test took cannot be replayed.
What the archive does give is the empirical distribution of baseline CTR, test size and (via
a proxy, see :func:`designs_from_rows`) the number of days a test spanned. The calibrated
functions here draw those three quantities jointly from real tests and then *simulate* the
daily stream. Every result from them is a calibrated simulation of how these rules would
have behaved on tests shaped like Upworthy's, not an observation of what Upworthy saw.

Everything here is pure numpy/scipy on arrays and returns frozen dataclasses.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import optimize, stats

__all__ = [
    "RULES",
    "Paths",
    "RuleOutcome",
    "SimulationResult",
    "TestDesign",
    "designs_from_rows",
    "draw_designs",
    "fixed_n",
    "look_fractions",
    "msprt",
    "msprt_p_values",
    "naive",
    "obf",
    "obf_boundaries",
    "obf_spending",
    "pooled_z",
    "simulate",
    "simulate_calibrated",
    "simulate_paths",
    "summarize",
]


# ---------------------------------------------------------------------------
# Streaming impressions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Paths:
    """Cumulative counts at each look for many simulated tests.

    Every array has shape ``(n_sims, n_looks)``; column k holds the totals after batch k.
    """

    n_c: np.ndarray
    x_c: np.ndarray
    n_t: np.ndarray
    x_t: np.ndarray

    @property
    def n_looks(self) -> int:
        return self.n_c.shape[1]

    @property
    def n_total(self) -> np.ndarray:
        """Impressions in both arms at the final look, per simulated test."""
        return self.n_c[:, -1] + self.n_t[:, -1]


def look_fractions(n_looks: int) -> np.ndarray:
    r"""Information fractions :math:`t_k = k / K` for K equal daily batches."""
    if n_looks < 1:
        raise ValueError(f"n_looks must be at least 1, got {n_looks}")
    return np.arange(1, n_looks + 1) / n_looks


def _cumulative_sizes(n_per_arm: np.ndarray, n_looks: int) -> np.ndarray:
    """Cumulative per-arm sample size after each of ``n_looks`` near-equal batches."""
    sizes = np.floor(np.outer(n_per_arm, look_fractions(n_looks))).astype(np.int64)
    sizes[:, -1] = n_per_arm
    return sizes


def simulate_paths(
    p_c: float | np.ndarray,
    lift: float,
    n_total: int | np.ndarray,
    n_looks: int,
    n_sims: int,
    rng: np.random.Generator,
) -> Paths:
    r"""Stream impressions for ``n_sims`` two-arm tests in ``n_looks`` daily batches.

    Each arm gets :math:`\lfloor n_{total}/2 \rfloor` impressions, delivered in equal batches
    (the last absorbs the rounding). Clicks in batch k are independent draws

    .. math::
        \Delta x_{c,k} \sim \text{Binomial}(\Delta n_k, p_c), \qquad
        \Delta x_{t,k} \sim \text{Binomial}(\Delta n_k, p_c (1 + \text{lift}))

    and the returned counts are cumulative sums. ``p_c`` and ``n_total`` may be scalars or
    arrays of length ``n_sims`` (one value per simulated test, as the calibrated simulation
    uses). ``lift`` is relative: 0 is the null, 0.1 a 10% higher treatment CTR.
    """
    if n_sims < 1:
        raise ValueError(f"n_sims must be at least 1, got {n_sims}")
    p_c = np.broadcast_to(np.asarray(p_c, dtype=float), (n_sims,))
    n_total = np.broadcast_to(np.asarray(n_total, dtype=np.int64), (n_sims,))
    p_t = p_c * (1 + lift)
    if np.any((p_c <= 0) | (p_c >= 1)) or np.any((p_t <= 0) | (p_t >= 1)):
        raise ValueError("p_c and p_c * (1 + lift) must be in (0, 1)")
    n_arm = n_total // 2
    if np.any(n_arm < n_looks):
        raise ValueError("each arm needs at least one impression per look")
    cum = _cumulative_sizes(n_arm, n_looks)
    batch = np.diff(cum, axis=1, prepend=0)
    x_c = rng.binomial(batch, p_c[:, None]).cumsum(axis=1)
    x_t = rng.binomial(batch, p_t[:, None]).cumsum(axis=1)
    return Paths(n_c=cum, x_c=x_c, n_t=cum.copy(), x_t=x_t)


def pooled_z(x_c: np.ndarray, n_c: np.ndarray, x_t: np.ndarray, n_t: np.ndarray) -> np.ndarray:
    r"""Vectorized pooled two-proportion z statistic (treatment minus control).

    .. math::
        z = \frac{\hat p_t - \hat p_c}{\sqrt{\hat p (1 - \hat p)(1/n_c + 1/n_t)}}

    Same statistic as :func:`abtest.proportions.two_proportion_ztest`; 0 where there is no
    variation yet (no clicks, or all clicks).
    """
    pooled = (x_c + x_t) / (n_c + n_t)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_t))
    diff = x_t / n_t - x_c / n_c
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(se > 0, diff / se, 0.0)


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleOutcome:
    """What a rule decided on each simulated test.

    ``stop_look`` is the 0-based look at which the rule rejected, or ``n_looks - 1`` if it
    never did (the test ran to the end). ``n_used`` is impressions in both arms at the stop.
    """

    rule: str
    reject: np.ndarray
    stop_look: np.ndarray
    n_used: np.ndarray


def _first_rejection(rule: str, paths: Paths, crossed: np.ndarray) -> RuleOutcome:
    reject = crossed.any(axis=1)
    stop = np.where(reject, crossed.argmax(axis=1), paths.n_looks - 1)
    rows = np.arange(len(stop))
    n_used = paths.n_c[rows, stop] + paths.n_t[rows, stop]
    return RuleOutcome(rule, reject, stop, n_used)


def naive(paths: Paths, alpha: float = 0.05) -> RuleOutcome:
    r"""Stop at the first look with a two-sided pooled z-test p-value below ``alpha``.

    Reject at look k if :math:`2\Phi(-|z_k|) < \alpha`, i.e. :math:`|z_k| > z_{1-\alpha/2}`
    at any k. With K looks the false positive rate is
    :math:`P(\max_k |Z_k| > 1.96) > \alpha` for K > 1, growing with K.
    """
    z = pooled_z(paths.x_c, paths.n_c, paths.x_t, paths.n_t)
    return _first_rejection("naive", paths, np.abs(z) > stats.norm.ppf(1 - alpha / 2))


def fixed_n(paths: Paths, alpha: float = 0.05) -> RuleOutcome:
    """The fixed-n reference: one z-test at the final look, nothing earlier."""
    z = pooled_z(paths.x_c[:, -1:], paths.n_c[:, -1:], paths.x_t[:, -1:], paths.n_t[:, -1:])
    crossed = np.zeros(paths.n_c.shape, dtype=bool)
    crossed[:, -1] = np.abs(z[:, 0]) > stats.norm.ppf(1 - alpha / 2)
    return _first_rejection("fixed", paths, crossed)


def obf_spending(t: float | np.ndarray, alpha: float = 0.05) -> np.ndarray:
    r"""Lan-DeMets O'Brien-Fleming-type spending, total over both sides of a symmetric test.

    Each side spends half of alpha with the one-sided Lan-DeMets function
    (the convention of ``gsDesign::sfLDOF`` and ``ldbounds``), so in total

    .. math::
        \alpha(t) = 2\left[2 - 2\,\Phi\!\left(\frac{z_{1-\alpha/4}}{\sqrt t}\right)\right],
        \quad 0 < t \le 1

    :math:`\alpha(1) = \alpha`, and almost nothing is spent early: at t = 0.2 it is about
    1e-6 for alpha = 0.05, which is why early OBF boundaries are so wide.
    """
    t = np.asarray(t, dtype=float)
    return 4 - 4 * stats.norm.cdf(stats.norm.ppf(1 - alpha / 4) / np.sqrt(t))


@lru_cache(maxsize=256)
def _obf_boundaries_cached(fractions: tuple[float, ...], alpha: float, grid: int) -> tuple:
    t = np.asarray(fractions)
    spent = obf_spending(t, alpha)
    increments = np.diff(spent, prepend=0.0)
    bounds: list[float] = []
    # Recursion on the Brownian motion B(t) = Z(t) sqrt(t) (Armitage, McPherson & Rowe 1969;
    # Jennison & Turnbull 2000, ch. 19): f holds the sub-density of B(t_k) on the
    # continuation region, evaluated on a grid with trapezoid weights.
    xs = w = f = None
    prev_t = 0.0
    for k, (tk, inc) in enumerate(zip(t, increments, strict=True)):
        sd = math.sqrt(tk - prev_t)
        if k == 0:
            c = float(stats.norm.isf(inc / 2)) if inc > 0 else math.inf
        else:

            def crossing(b, xs=xs, w=w, f=f, sd=sd):
                tail = stats.norm.sf((b - xs) / sd) + stats.norm.cdf((-b - xs) / sd)
                return float(np.sum(w * f * tail))

            if inc <= 1e-15:
                c = math.inf
            else:
                upper = 40.0 * math.sqrt(tk)
                b = optimize.brentq(lambda b, inc=inc: crossing(b) - inc, 1e-9, upper, xtol=1e-12)
                c = b / math.sqrt(tk)
        bounds.append(c)
        if k == len(t) - 1:
            break
        # Density of B(t_k) restricted to |B| < b_k (cap an infinite bound at 10 sd).
        b_k = c * math.sqrt(tk) if math.isfinite(c) else 10.0 * math.sqrt(tk)
        new_xs = np.linspace(-b_k, b_k, grid)
        new_w = np.full(grid, new_xs[1] - new_xs[0])
        new_w[[0, -1]] /= 2
        if k == 0:
            new_f = stats.norm.pdf(new_xs, scale=sd)
        else:
            kern = stats.norm.pdf((new_xs[:, None] - xs[None, :]) / sd) / sd
            new_f = kern @ (w * f)
        xs, w, f = new_xs, new_w, new_f
        prev_t = tk
    return tuple(bounds)


def obf_boundaries(
    fractions: Sequence[float] | np.ndarray, alpha: float = 0.05, grid: int = 801
) -> np.ndarray:
    r"""Two-sided z boundaries :math:`c_k` for Lan-DeMets O'Brien-Fleming spending.

    With information fractions :math:`0 < t_1 < \dots < t_K = 1`, each :math:`c_k` solves

    .. math::
        P_0\big(|Z_1| < c_1, \dots, |Z_{k-1}| < c_{k-1}, |Z_k| \ge c_k\big)
        = \alpha(t_k) - \alpha(t_{k-1})

    where :math:`Z_k = B(t_k)/\sqrt{t_k}` for a standard Brownian motion B, so
    :math:`\mathrm{corr}(Z_i, Z_j) = \sqrt{t_i / t_j}`. The probabilities come from numerical
    integration of B's density over the continuation region (``grid`` points per look).
    For five equal looks at alpha = 0.05 this gives 4.877, 3.357, 2.680, 2.290, 2.031,
    the textbook values. ``inf`` means no rejection is possible at that look.
    """
    t = np.asarray(fractions, dtype=float)
    if t.ndim != 1 or len(t) == 0:
        raise ValueError("fractions must be a non-empty 1-d sequence")
    if np.any(np.diff(t) <= 0) or t[0] <= 0 or not math.isclose(t[-1], 1.0):
        raise ValueError("fractions must increase strictly and end at 1")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    key = tuple(round(float(x), 12) for x in t)
    return np.asarray(_obf_boundaries_cached(key, float(alpha), int(grid)))


def obf(paths: Paths, alpha: float = 0.05) -> RuleOutcome:
    r"""Group-sequential test with Lan-DeMets O'Brien-Fleming alpha spending.

    Reject at the first look k with :math:`|z_k| \ge c_k` (see :func:`obf_boundaries`),
    using the planned equal-batch fractions :math:`t_k = k/K` and the pooled z statistic.
    Total type I error is alpha; the last boundary is only a little above 1.96, so power is
    close to the fixed-n test while early stopping saves sample when the effect is large.
    """
    c = obf_boundaries(look_fractions(paths.n_looks), alpha)
    z = pooled_z(paths.x_c, paths.n_c, paths.x_t, paths.n_t)
    return _first_rejection("obf", paths, np.abs(z) >= c[None, :])


def msprt_p_values(paths: Paths, tau2: float | np.ndarray) -> np.ndarray:
    r"""Always-valid mSPRT p-values at each look (Johari et al. 2017, normal mixture).

    Let :math:`\hat\theta_k = \hat p_{t,k} - \hat p_{c,k}` with estimated variance
    :math:`V_k = \hat p_{c,k}(1-\hat p_{c,k})/n_{c,k} + \hat p_{t,k}(1-\hat p_{t,k})/n_{t,k}`.
    Treating :math:`\hat\theta_k \sim N(\theta, V_k)` and mixing the alternative over
    :math:`\theta \sim N(0, \tau^2)`, the likelihood ratio against :math:`\theta = 0` is

    .. math::
        \Lambda_k = \sqrt{\frac{V_k}{V_k + \tau^2}}
        \exp\!\left(\frac{\tau^2 \hat\theta_k^2}{2 V_k (V_k + \tau^2)}\right)

    and the always-valid p-value is :math:`p_k = \min(1, p_{k-1}, 1/\Lambda_k)`, with
    :math:`p_0 = 1`. ``tau2`` is the mixing variance on the absolute difference in rates
    (a scalar, or one value per simulated test); it should be on the scale of the effects
    you expect, e.g. ``(p_c * 0.1) ** 2`` for lifts around 10%. Where :math:`V_k = 0` (no
    clicks yet) :math:`\Lambda_k = 1`.
    """
    tau2 = np.asarray(tau2, dtype=float)
    if np.any(tau2 <= 0):
        raise ValueError("tau2 must be positive")
    if tau2.ndim == 1:
        tau2 = tau2[:, None]
    pc = paths.x_c / paths.n_c
    pt = paths.x_t / paths.n_t
    v = pc * (1 - pc) / paths.n_c + pt * (1 - pt) / paths.n_t
    theta = pt - pc
    with np.errstate(divide="ignore", invalid="ignore"):
        log_lr = 0.5 * np.log(v / (v + tau2)) + tau2 * theta**2 / (2 * v * (v + tau2))
    log_lr = np.where(v > 0, log_lr, 0.0)
    p = np.minimum(1.0, np.exp(-log_lr))
    return np.minimum.accumulate(p, axis=1)


def msprt(paths: Paths, alpha: float = 0.05, tau2: float | np.ndarray = 1e-6) -> RuleOutcome:
    """Stop at the first look where the always-valid mSPRT p-value drops below ``alpha``.

    See :func:`msprt_p_values` for the statistic and the meaning of ``tau2``.
    """
    return _first_rejection("msprt", paths, msprt_p_values(paths, tau2) < alpha)


RULES: dict[str, Callable[..., RuleOutcome]] = {
    "fixed": fixed_n,
    "naive": naive,
    "obf": obf,
    "msprt": msprt,
}


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationResult:
    r"""Operating characteristics of one rule over many simulated tests.

    ``rejection_rate`` is the false positive rate when ``lift == 0`` and the power otherwise,
    with binomial Monte Carlo standard error :math:`\sqrt{r(1-r)/n_{sims}}`.
    ``sample_saved`` is :math:`1 - E[n_{used}] / E[n_{total}]`, the share of the fixed-n
    sample the rule did not need on average (0 for the fixed-n test by construction).
    """

    rule: str
    n_looks: int | None
    lift: float
    n_sims: int
    rejection_rate: float
    se: float
    mean_n_used: float
    mean_n_total: float
    sample_saved: float


def summarize(outcome: RuleOutcome, paths: Paths, lift: float) -> SimulationResult:
    """Collapse a :class:`RuleOutcome` into a :class:`SimulationResult`."""
    n = len(outcome.reject)
    r = float(outcome.reject.mean())
    n_total = float(paths.n_total.mean())
    used = float(outcome.n_used.mean())
    return SimulationResult(
        rule=outcome.rule,
        n_looks=paths.n_looks,
        lift=lift,
        n_sims=n,
        rejection_rate=r,
        se=math.sqrt(r * (1 - r) / n),
        mean_n_used=used,
        mean_n_total=n_total,
        sample_saved=1 - used / n_total,
    )


def _apply(rule: str, paths: Paths, alpha: float, tau2) -> RuleOutcome:
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; choose from {sorted(RULES)}")
    if rule == "msprt":
        return msprt(paths, alpha, tau2)
    return RULES[rule](paths, alpha)


def simulate(
    p_c: float,
    lift: float,
    n_total: int,
    n_looks: int,
    rules: Iterable[str] = ("fixed", "naive", "obf", "msprt"),
    n_sims: int = 10_000,
    alpha: float = 0.05,
    tau2: float | None = None,
    seed: int | np.random.Generator | None = 0,
) -> dict[str, SimulationResult]:
    """Simulate one fixed design and score every rule on the **same** simulated paths.

    ``tau2`` defaults to ``(0.1 * p_c) ** 2``: a mixing distribution centred on relative
    lifts of about 10%, the scale of a good headline test.
    """
    rng = np.random.default_rng(seed)
    tau2 = (0.1 * p_c) ** 2 if tau2 is None else tau2
    paths = simulate_paths(p_c, lift, n_total, n_looks, n_sims, rng)
    return {r: summarize(_apply(r, paths, alpha, tau2), paths, lift) for r in rules}


# ---------------------------------------------------------------------------
# Calibration to the archive
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestDesign:
    """The shape of one real test, as used to calibrate a simulated two-arm test.

    ``baseline_ctr`` is the test's pooled CTR (all clicks over all impressions),
    ``n_total`` the impressions a two-arm comparison in that test had (two arms' worth:
    2 x impressions per arm), and ``n_looks`` the number of daily looks (see
    :func:`designs_from_rows`).
    """

    __test__ = False  # not a pytest test class

    test_id: str
    baseline_ctr: float
    n_total: int
    n_looks: int


def designs_from_rows(
    rows: Iterable[tuple[str, int, int, int, int]], min_clicks: int = 1
) -> list[TestDesign]:
    """Build :class:`TestDesign` records from per-test rows of the archive.

    Each row is ``(test_id, n_arms, total_impressions, total_clicks, days_spanned)``, where
    ``days_spanned`` is the number of calendar days from the first to the last arm's
    ``created_at``, inclusive (so a test whose arms were all created on one day has 1).

    **How the number of looks is derived.** The archive has no end date and no daily
    counts, so test duration is not observed. The only time information per arm is
    ``created_at`` (when the package was created) plus one ``test_week`` per test.
    ``n_looks = days_spanned`` is the proxy used here: one look per calendar day on which
    the test was visibly being worked on. It is a lower bound on the days a test ran, and in
    the archive most tests have all arms created within minutes, so most tests get 1 look.
    The sweep over a fixed number of looks in ``analysis/peeking.py`` does not depend on it.

    Single-arm tests, tests with fewer than ``min_clicks`` clicks (no usable baseline) and
    tests too small to give each arm one impression per look are dropped.
    """
    out = []
    for test_id, n_arms, impressions, clicks, days in rows:
        if n_arms < 2 or impressions <= 0 or clicks < min_clicks or clicks >= impressions:
            continue
        looks = max(1, int(days))
        n_total = 2 * (int(impressions) // int(n_arms))
        if n_total // 2 < looks:
            continue
        out.append(TestDesign(str(test_id), clicks / impressions, n_total, looks))
    return out


def draw_designs(
    designs: Sequence[TestDesign], size: int, rng: np.random.Generator
) -> list[TestDesign]:
    """Resample whole tests with replacement, keeping CTR, n and looks jointly as observed."""
    if not designs:
        raise ValueError("no designs to draw from")
    idx = rng.integers(0, len(designs), size=size)
    return [designs[i] for i in idx]


def simulate_calibrated(
    designs: Sequence[TestDesign],
    lift: float,
    rules: Iterable[str] = ("fixed", "naive", "obf", "msprt"),
    n_sims: int = 10_000,
    n_looks: int | None = None,
    alpha: float = 0.05,
    tau_rel: float = 0.1,
    seed: int | np.random.Generator | None = 0,
) -> dict[str, SimulationResult]:
    r"""Score the rules on simulated tests whose shape is drawn from real ones.

    Each simulated test draws one real :class:`TestDesign` (baseline CTR, n and looks, jointly)
    and streams impressions with :func:`simulate_paths`. With ``n_looks`` given, every test
    uses that many looks instead of its own (for a sweep over looks with CTR and n still drawn
    from the data). The mSPRT mixing variance is set per test as
    :math:`\tau^2 = (\tau_{rel}\, p_c)^2`.

    This is a calibrated simulation: the archive has no daily counts, so the daily stream is
    simulated, not observed. Results with ``n_looks=None`` mix tests with different numbers
    of looks; their ``n_looks`` field is None.
    """
    rng = np.random.default_rng(seed)
    rules = tuple(rules)
    drawn = draw_designs(designs, n_sims, rng)
    looks = np.array([n_looks or d.n_looks for d in drawn])
    p_c = np.array([d.baseline_ctr for d in drawn])
    n_tot = np.array([d.n_total for d in drawn])
    keep = n_tot // 2 >= looks
    if not keep.any():
        raise ValueError("no drawn design is large enough for the requested number of looks")
    looks, p_c, n_tot = looks[keep], p_c[keep], n_tot[keep]
    # Keep only designs where the treatment rate stays a valid probability.
    ok = p_c * (1 + lift) < 1
    looks, p_c, n_tot = looks[ok], p_c[ok], n_tot[ok]

    acc: dict[str, dict[str, float]] = {
        r: {"reject": 0.0, "used": 0.0, "total": 0.0, "n": 0} for r in rules
    }
    for k in np.unique(looks):
        m = looks == k
        paths = simulate_paths(p_c[m], lift, n_tot[m], int(k), int(m.sum()), rng)
        tau2 = (tau_rel * p_c[m]) ** 2
        for r in rules:
            o = _apply(r, paths, alpha, tau2)
            a = acc[r]
            a["reject"] += float(o.reject.sum())
            a["used"] += float(o.n_used.sum())
            a["total"] += float(paths.n_total.sum())
            a["n"] += int(m.sum())

    out = {}
    for r, a in acc.items():
        n = a["n"]
        rate = a["reject"] / n
        out[r] = SimulationResult(
            rule=r,
            n_looks=n_looks,
            lift=lift,
            n_sims=n,
            rejection_rate=rate,
            se=math.sqrt(rate * (1 - rate) / n),
            mean_n_used=a["used"] / n,
            mean_n_total=a["total"] / n,
            sample_saved=1 - a["used"] / a["total"],
        )
    return out
