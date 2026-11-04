r"""Bayesian tools: beta-binomial arm posteriors, and empirical Bayes shrinkage across tests.

Two levels, like :mod:`abtest.multiple`:

* **inside one test**: each arm's click rate gets a Beta posterior
  (:func:`beta_posteriors`); Monte Carlo draws from those posteriors give the chance each
  arm is the best and the expected loss of shipping it (:func:`prob_best`), and
  :func:`decide` ships the arm with the smallest expected loss when that loss is below a
  threshold.
* **across the portfolio**: every test's lift is measured on the log-odds scale with a
  standard error (:func:`arm_contrasts`, :func:`best_arm_contrast`), a prior for the true
  lifts is fitted from all tests at once (:func:`fit_normal_prior` by method of moments,
  :func:`fit_spike_slab` by maximum likelihood), and each observed lift is pulled toward
  that prior (:func:`shrink`). :func:`shrink_best_arms` does this for the arm each test
  would pick as its winner, and :func:`winners_curse` compares the naive and shrunk lifts.

Why shrinkage fixes the winner's curse
--------------------------------------
Every observed lift is the true lift plus noise. When a test picks the arm with the highest
observed click rate, it tends to pick an arm whose noise happened to be positive, because a
lucky draw is one of the ways an arm gets to the top. So the winner's observed lift is, on
average, bigger than its true lift, and the noisier the test the bigger the gap. That gap is
the winner's curse, and it is why winners usually do worse once rolled out.

Shrinkage asks a different question: given everything we have seen across thousands of
tests, what true lift is most likely to produce this observed lift? Across the archive,
true lifts turn out to be small and tightly bunched near zero, while observed lifts are
spread wide by noise. So a big observed lift from a noisy test is far more often a small
true lift plus lucky noise than a genuinely big lift, and the posterior mean pulls it most of
the way back. A precise test (small standard error) is pulled back only a little, because
its noise cannot explain much. The pull depends only on the observed lift and its noise,
not on whether the arm was picked, so it removes on average exactly the luck that picking the
maximum selected for: the posterior mean of the picked arm is an honest forecast of what it
will do next.

One catch: the prior must describe arms *before* anyone picked the best one. Fitted on
winners alone it would learn the inflated lifts as if they were real and shrink toward them.
So by default the prior is fitted on every arm against the rest of its test (selection
free), and then applied to the arm that was picked.

Notation: for one arm with :math:`x` clicks in :math:`n` impressions against the pooled
other arms of its test (:math:`x_r`, :math:`n_r`), the observed lift is the log odds ratio
:math:`d = \operatorname{logit}(x/n) - \operatorname{logit}(x_r/n_r)` with standard error
:math:`s = \sqrt{1/x + 1/(n-x) + 1/x_r + 1/(n_r-x_r)}`, and 0.5 is added to all four cells
when any is zero. For click rates of a few percent the odds ratio is close to the rate
ratio, so :math:`e^d - 1` reads as a relative lift.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import optimize, stats

__all__ = [
    "ArmPosterior",
    "BestArmShrinkage",
    "Contrast",
    "Decision",
    "NormalPrior",
    "ProbBest",
    "Shrunk",
    "SpikeSlabPrior",
    "WinnersCurse",
    "arm_contrasts",
    "best_arm_contrast",
    "beta_posteriors",
    "decide",
    "fit_normal_prior",
    "fit_spike_slab",
    "normal_loglik",
    "prob_best",
    "shrink",
    "shrink_best_arms",
    "winners_curse",
]

Z95 = float(stats.norm.ppf(0.975))


def _counts(clicks: Sequence[int], impressions: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(clicks, dtype=float)
    n = np.asarray(impressions, dtype=float)
    if x.ndim != 1 or x.shape != n.shape or x.size == 0:
        raise ValueError("clicks and impressions must be equal-length, non-empty 1-d sequences")
    if (x < 0).any() or (x > n).any():
        raise ValueError("need 0 <= clicks <= impressions for every arm")
    return x, n


# --- beta-binomial, one test -----------------------------------------------------------------


@dataclass(frozen=True)
class ArmPosterior:
    """Beta(alpha, beta) posterior for one arm's click rate, with an equal-tailed interval."""

    alpha: float
    beta: float
    mean: float
    lower: float
    upper: float


def beta_posteriors(
    clicks: Sequence[int],
    impressions: Sequence[int],
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    level: float = 0.95,
) -> list[ArmPosterior]:
    """Conjugate posterior for each arm: Beta(prior_alpha + x, prior_beta + n - x).

    The default Beta(1, 1) prior is uniform on the click rate. ``level`` sets the
    equal-tailed credible interval.
    """
    x, n = _counts(clicks, impressions)
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("prior_alpha and prior_beta must be positive")
    a, b = prior_alpha + x, prior_beta + n - x
    tail = (1 - level) / 2
    lo, hi = stats.beta.ppf(tail, a, b), stats.beta.ppf(1 - tail, a, b)
    return [
        ArmPosterior(float(ai), float(bi), float(ai / (ai + bi)), float(li), float(ui))
        for ai, bi, li, ui in zip(a, b, lo, hi, strict=True)
    ]


@dataclass(frozen=True)
class ProbBest:
    """Monte Carlo summary of which arm is best.

    ``prob_best[i]`` is P(arm i has the highest click rate); the entries sum to 1.
    ``expected_loss[i]`` is E[max_j p_j - p_i], the click rate given up on average by
    shipping arm i (absolute, in click-rate units). ``mean[i]`` is the posterior mean.
    """

    prob_best: tuple[float, ...]
    expected_loss: tuple[float, ...]
    mean: tuple[float, ...]
    n_draws: int
    seed: int | None


def prob_best(
    clicks: Sequence[int],
    impressions: Sequence[int],
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_draws: int = 20_000,
    seed: int | None = 0,
) -> ProbBest:
    """P(best) and expected loss for each arm, from ``n_draws`` joint posterior draws.

    Draws are reproducible for a fixed ``seed``. Ties within a draw go to the lower index,
    which has probability zero for continuous posteriors but keeps the sum at exactly 1.
    """
    x, n = _counts(clicks, impressions)
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    posts = beta_posteriors(x, n, prior_alpha, prior_beta)
    rng = np.random.default_rng(seed)
    a = np.array([p.alpha for p in posts])[:, None]
    b = np.array([p.beta for p in posts])[:, None]
    draws = rng.beta(a, b, size=(len(posts), n_draws))
    best = draws.argmax(axis=0)
    pb = np.bincount(best, minlength=len(posts)) / n_draws
    loss = (draws.max(axis=0) - draws).mean(axis=1)
    return ProbBest(
        tuple(float(v) for v in pb),
        tuple(float(v) for v in loss),
        tuple(p.mean for p in posts),
        n_draws,
        seed,
    )


@dataclass(frozen=True)
class Decision:
    """The arm with the smallest expected loss, and whether to ship it.

    ``loss`` is in the units the threshold was compared in: absolute click rate, or a
    fraction of the chosen arm's posterior mean when ``relative`` is true.
    """

    arm: int
    prob_best: float
    loss: float
    threshold: float
    relative: bool
    ship: bool


def decide(result: ProbBest, threshold: float, relative: bool = False) -> Decision:
    """Ship the minimum-expected-loss arm if its expected loss is below ``threshold``.

    With ``relative=True`` the loss is divided by the arm's posterior mean, so
    ``threshold=0.01`` means "we give up less than 1% of the arm's click rate on average".
    Otherwise the threshold is in absolute click-rate units. The rule keeps testing (does
    not ship) when no arm is clearly safe, which is the honest call for an underpowered test.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    losses = np.asarray(result.expected_loss)
    arm = int(losses.argmin())
    loss = float(losses[arm])
    if relative:
        loss = loss / result.mean[arm] if result.mean[arm] > 0 else float("inf")
    return Decision(arm, result.prob_best[arm], loss, threshold, relative, loss < threshold)


# --- contrasts on the log-odds scale ---------------------------------------------------------


@dataclass(frozen=True)
class Contrast:
    """One arm against the pooled rest of its test: log odds ratio ``d`` and its ``se``."""

    arm: int
    d: float
    se: float


def _log_or(x: float, n: float, xr: float, nr: float) -> tuple[float, float]:
    cells = np.array([x, n - x, xr, nr - xr], dtype=float)
    if (cells == 0).any():
        cells = cells + 0.5
    a, b, c, e = cells
    return float(np.log(a / b) - np.log(c / e)), float(np.sqrt((1 / cells).sum()))


def arm_contrasts(clicks: Sequence[int], impressions: Sequence[int]) -> list[Contrast]:
    """Every arm against the pooled other arms of its test (arms with no impressions dropped).

    Returns an empty list when fewer than two arms have impressions.
    """
    x, n = _counts(clicks, impressions)
    keep = np.flatnonzero(n > 0)
    if keep.size < 2:
        return []
    tx, tn = x[keep].sum(), n[keep].sum()
    return [Contrast(int(i), *_log_or(x[i], n[i], tx - x[i], tn - n[i])) for i in keep]


def best_arm_contrast(clicks: Sequence[int], impressions: Sequence[int]) -> Contrast | None:
    """The arm a naive analyst would ship (highest observed click rate) against the rest.

    Ties go to the lower index. ``None`` when fewer than two arms have impressions.
    """
    x, n = _counts(clicks, impressions)
    rate = np.where(n > 0, x / np.where(n > 0, n, 1), -np.inf)
    if (n > 0).sum() < 2:
        return None
    best = int(rate.argmax())
    return next(c for c in arm_contrasts(x, n) if c.arm == best)


# --- priors over true lifts ------------------------------------------------------------------


def _lifts(d: Sequence[float] | np.ndarray, se: Sequence[float] | np.ndarray):
    d = np.asarray(d, dtype=float)
    se = np.asarray(se, dtype=float)
    if d.ndim != 1 or d.shape != se.shape or d.size < 2:
        raise ValueError("d and se must be equal-length 1-d sequences with 2 or more entries")
    if not (np.isfinite(d).all() and np.isfinite(se).all()) or (se <= 0).any():
        raise ValueError("d must be finite and se finite and positive")
    return d, se


@dataclass(frozen=True)
class NormalPrior:
    """True lifts ~ Normal(mu, tau^2)."""

    mu: float
    tau2: float
    n: int

    @property
    def tau(self) -> float:
        return float(np.sqrt(self.tau2))


@dataclass(frozen=True)
class SpikeSlabPrior:
    """True lifts ~ pi0 * (point mass at 0) + (1 - pi0) * Normal(mu, tau^2).

    ``pi0`` is the share of arms with no true lift at all. ``loglik`` is the maximized
    marginal log likelihood, comparable with :meth:`NormalPrior` fits via
    :func:`normal_loglik`.
    """

    pi0: float
    mu: float
    tau2: float
    loglik: float
    n: int

    @property
    def tau(self) -> float:
        return float(np.sqrt(self.tau2))


def fit_normal_prior(d: Sequence[float], se: Sequence[float]) -> NormalPrior:
    r"""Normal prior by method of moments.

    :math:`\hat\mu = \bar d` and :math:`\hat\tau^2 = \max(0, s_d^2 - \overline{s^2})`.

    Each observed lift is the true lift plus independent noise with known variance
    :math:`s_i^2`, so the variance of the observed lifts is the prior variance plus the
    average noise variance. Subtracting the noise leaves the spread of the true lifts.
    """
    d, se = _lifts(d, se)
    return NormalPrior(
        float(d.mean()), float(max(0.0, d.var(ddof=1) - (se**2).mean())), int(d.size)
    )


def _spike_slab_nll(params: np.ndarray, d: np.ndarray, s2: np.ndarray) -> float:
    logit_pi, mu, log_tau = params
    pi0 = 1 / (1 + np.exp(-logit_pi))
    tau2 = np.exp(2 * log_tau)
    spike = stats.norm.logpdf(d, 0.0, np.sqrt(s2))
    slab = stats.norm.logpdf(d, mu, np.sqrt(s2 + tau2))
    ll = np.logaddexp(np.log(pi0) + spike, np.log1p(-pi0) + slab)
    return float(-ll.sum())


def normal_loglik(prior: NormalPrior, d: Sequence[float], se: Sequence[float]) -> float:
    """Marginal log likelihood of the observed lifts under a normal prior."""
    d, se = _lifts(d, se)
    return float(stats.norm.logpdf(d, prior.mu, np.sqrt(se**2 + prior.tau2)).sum())


def fit_spike_slab(d: Sequence[float], se: Sequence[float]) -> SpikeSlabPrior:
    """Maximum likelihood fit of the spike-at-zero plus normal prior.

    The marginal of each observed lift is
    ``pi0 * N(0, s_i^2) + (1 - pi0) * N(mu, tau^2 + s_i^2)``. The fit starts from the method
    of moments normal prior at several values of ``pi0`` and keeps the best optimum.
    """
    d, se = _lifts(d, se)
    s2 = se**2
    mom = fit_normal_prior(d, se)
    tau0 = max(mom.tau, 0.5 * float(np.sqrt(s2.mean())), 1e-3)
    best = None
    for pi0 in (0.2, 0.5, 0.8):
        start = np.array([np.log(pi0 / (1 - pi0)), mom.mu, np.log(tau0 / np.sqrt(1 - pi0))])
        res = optimize.minimize(
            _spike_slab_nll, start, args=(d, s2), method="L-BFGS-B",
            bounds=[(-12, 12), (None, None), (-12, 5)],
        )  # fmt: skip
        if best is None or res.fun < best.fun:
            best = res
    logit_pi, mu, log_tau = best.x
    return SpikeSlabPrior(
        float(1 / (1 + np.exp(-logit_pi))),
        float(mu),
        float(np.exp(2 * log_tau)),
        float(-best.fun),
        int(d.size),
    )


# --- posteriors ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Shrunk:
    """Posterior summaries of each true lift (log-odds scale), aligned with the input.

    ``p_null`` is the posterior probability of exactly zero lift (all zeros under a normal
    prior). ``lower``/``upper`` bound a 95% equal-tailed credible interval.
    """

    d: np.ndarray
    se: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    p_null: np.ndarray


def _mixture_quantile(q: float, w: np.ndarray, m: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Quantile of w * delta_0 + (1 - w) * N(m, sd^2), by vectorized bisection."""
    sd = np.maximum(sd, 1e-12)
    lo = np.minimum(0.0, m - 12 * sd) - 1e-9
    hi = np.maximum(0.0, m + 12 * sd) + 1e-9
    for _ in range(80):
        mid = (lo + hi) / 2
        cdf = w * (mid >= 0) + (1 - w) * stats.norm.cdf((mid - m) / sd)
        go_left = cdf >= q
        hi = np.where(go_left, mid, hi)
        lo = np.where(go_left, lo, mid)
    return hi


def shrink(prior: NormalPrior | SpikeSlabPrior, d: Sequence[float], se: Sequence[float]) -> Shrunk:
    r"""Posterior mean and 95% interval of each true lift given its observed ``d`` and ``se``.

    Normal prior: the posterior is normal with mean :math:`\mu + B_i (d_i - \mu)`, where
    :math:`B_i = \tau^2 / (\tau^2 + s_i^2)` is the share of the observed deviation that is
    believed. Noisy lifts (large :math:`s_i`) get small :math:`B_i` and are pulled hard
    toward :math:`\mu`; precise lifts keep most of their value.

    Spike and slab: with probability ``p_null`` the lift is exactly zero, otherwise it is
    normal as above with the slab's mean and variance; the posterior mean is
    ``(1 - p_null)`` times the slab mean and the interval comes from the mixture's quantiles.
    """
    d, se = _lifts(d, se)
    s2 = se**2
    if isinstance(prior, SpikeSlabPrior):
        log_spike = np.log(prior.pi0) + stats.norm.logpdf(d, 0.0, se)
        log_slab = np.log1p(-prior.pi0) + stats.norm.logpdf(d, prior.mu, np.sqrt(s2 + prior.tau2))
        w = np.exp(log_spike - np.logaddexp(log_spike, log_slab))
    else:
        w = np.zeros_like(d)
    b = prior.tau2 / (prior.tau2 + s2)
    m = prior.mu + b * (d - prior.mu)
    v = b * s2
    mean = (1 - w) * m
    var = (1 - w) * (v + m**2) - mean**2
    sd = np.sqrt(np.maximum(var, 0.0))
    if isinstance(prior, SpikeSlabPrior):
        lower = _mixture_quantile(0.025, w, m, np.sqrt(v))
        upper = _mixture_quantile(0.975, w, m, np.sqrt(v))
    else:
        lower, upper = m - Z95 * np.sqrt(v), m + Z95 * np.sqrt(v)
    return Shrunk(d, se, mean, sd, lower, upper, w)


# --- across tests: the arm each test picks ---------------------------------------------------


@dataclass(frozen=True)
class WinnersCurse:
    """How much of the picked arms' observed lift survives shrinkage (log-odds scale).

    ``retained = mean_shrunk / mean_naive`` is the share of the naive average lift that is
    expected to be real; ``1 - retained`` is the average inflation from picking the max.
    The ``*_rel`` fields convert means to relative lift, ``exp(x) - 1``.
    """

    n_tests: int
    mean_naive: float
    mean_shrunk: float
    median_naive: float
    median_shrunk: float
    retained: float
    mean_naive_rel: float
    mean_shrunk_rel: float
    share_interval_excludes_zero_naive: float
    share_interval_excludes_zero_shrunk: float


@dataclass(frozen=True)
class BestArmShrinkage:
    """Per-test naive and shrunk lifts of the picked arm, with the prior that did it.

    ``test_index[i]`` points back into the tests passed in (tests with fewer than two arms
    with impressions are skipped). ``fit_on`` says which contrasts fitted the prior.
    """

    test_index: np.ndarray
    arm: np.ndarray
    shrunk: Shrunk
    prior: NormalPrior | SpikeSlabPrior
    fit_on: str
    n_fit: int


def shrink_best_arms(
    tests: Iterable[tuple[Sequence[int], Sequence[int]]],
    prior: Literal["normal", "spike_slab"] = "spike_slab",
    fit_on: Literal["all", "best"] = "all",
) -> BestArmShrinkage:
    """Empirical Bayes posterior for each test's picked arm.

    ``tests`` yields ``(clicks, impressions)`` per test. The picked arm is the one with the
    highest observed click rate, and its lift is against the pooled rest of its test.

    ``fit_on="all"`` (the default) fits the prior on every arm's contrast against the rest of
    its test, before any picking, which is the population the picked arm was drawn from.
    Contrasts within a test share arms, so this is a composite likelihood: the estimates are
    consistent but their nominal precision is overstated. ``fit_on="best"`` fits on the
    picked arms only; it is there to show why that is wrong (the prior absorbs the winner's
    curse and shrinks toward the inflated average).
    """
    picked, idx, arms, fit_d, fit_se = [], [], [], [], []
    for i, (clicks, imps) in enumerate(tests):
        best = best_arm_contrast(clicks, imps)
        if best is None:
            continue
        picked.append(best)
        idx.append(i)
        arms.append(best.arm)
        if fit_on == "all":
            for c in arm_contrasts(clicks, imps):
                fit_d.append(c.d)
                fit_se.append(c.se)
    if fit_on == "best":
        fit_d = [c.d for c in picked]
        fit_se = [c.se for c in picked]
    elif fit_on != "all":
        raise ValueError(f"fit_on must be 'all' or 'best', got {fit_on!r}")
    if len(picked) < 2:
        raise ValueError("need at least two tests with 2 or more arms")
    if prior == "normal":
        fitted: NormalPrior | SpikeSlabPrior = fit_normal_prior(fit_d, fit_se)
    elif prior == "spike_slab":
        fitted = fit_spike_slab(fit_d, fit_se)
    else:
        raise ValueError(f"prior must be 'normal' or 'spike_slab', got {prior!r}")
    post = shrink(fitted, [c.d for c in picked], [c.se for c in picked])
    return BestArmShrinkage(np.array(idx), np.array(arms), post, fitted, fit_on, len(fit_d))


def winners_curse(result: BestArmShrinkage | Shrunk) -> WinnersCurse:
    """Compare the picked arms' naive observed lifts with their shrunk posterior means.

    The naive lift is what a dashboard reports for the winner; the shrunk lift is the
    expected true lift given the data and the fitted prior. Their ratio is the share of the
    reported lift you should expect to see after rollout.
    """
    s = result.shrunk if isinstance(result, BestArmShrinkage) else result
    naive_lo, naive_hi = s.d - Z95 * s.se, s.d + Z95 * s.se
    mean_naive, mean_shrunk = float(s.d.mean()), float(s.mean.mean())
    return WinnersCurse(
        n_tests=int(s.d.size),
        mean_naive=mean_naive,
        mean_shrunk=mean_shrunk,
        median_naive=float(np.median(s.d)),
        median_shrunk=float(np.median(s.mean)),
        retained=mean_shrunk / mean_naive if mean_naive != 0 else float("nan"),
        mean_naive_rel=float(np.expm1(mean_naive)),
        mean_shrunk_rel=float(np.expm1(mean_shrunk)),
        share_interval_excludes_zero_naive=float(((naive_lo > 0) | (naive_hi < 0)).mean()),
        share_interval_excludes_zero_shrunk=float(((s.lower > 0) | (s.upper < 0)).mean()),
    )
