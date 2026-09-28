"""
Uncertainty and paired comparisons over structure-level units.

  * ci():          95% CI of a condition's mean -- Wilson for 0/1 outcomes
                   (bootstrap collapses to [p, p] at 0% or 100%), otherwise a
                   percentile bootstrap over structures.
  * paired_diff(): difference vs a reference on the structures both ran,
                   with a bootstrap CI and a sign-flip permutation p-value
                   (exact up to 20 structures, Monte Carlo beyond). No
                   normality assumption, which matters for bounded rates.
  * holm():        family-wise correction across the metrics compared.
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

N_BOOT = 10_000
ALPHA = 0.05


@dataclass
class Estimate:
    mean: float
    lo: float
    hi: float
    n: int

    @property
    def half_width(self):
        return (self.hi - self.lo) / 2


@dataclass
class PairedDiff:
    diff: float
    lo: float
    hi: float
    p: float
    n: int
    p_holm: float = float("nan")


def ci(values, binary=False, seed=0):
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return Estimate(np.nan, np.nan, np.nan, 0)
    mean = float(x.mean())
    if n == 1:
        return Estimate(mean, np.nan, np.nan, 1)
    # Wilson only for genuinely 0/1 outcomes (one per structure) -- a rate
    # metric that happens to be all zeros must not get a binomial interval
    if binary and np.isin(x, (0.0, 1.0)).all():
        z = sps.norm.ppf(1 - ALPHA / 2)
        denom = 1 + z**2 / n
        centre = (mean + z**2 / (2 * n)) / denom
        half = z * np.sqrt(mean * (1 - mean) / n + z**2 / (4 * n**2)) / denom
        return Estimate(mean, centre - half, centre + half, n)
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    lo, hi = np.quantile(boots, [ALPHA / 2, 1 - ALPHA / 2])
    return Estimate(mean, float(lo), float(hi), n)


def paired_diff(a, b, seed=0):
    """a, b: pd.Series indexed by structure_idx. Returns a - b."""
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).to_numpy(dtype=float)
    n = len(d)
    if n < 2:
        return PairedDiff(float(d.mean()) if n else np.nan, np.nan, np.nan, np.nan, n)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    lo, hi = np.quantile(boots, [ALPHA / 2, 1 - ALPHA / 2])

    observed = abs(d.mean()) - 1e-12
    bits = np.arange(n)
    if n <= 20:
        hits, total, chunk = 0, 2**n, 2**16
        for start in range(0, total, chunk):
            codes = np.arange(start, min(start + chunk, total))[:, None]
            signs = 1.0 - 2.0 * ((codes >> bits) & 1)
            hits += int((np.abs(signs @ d) / n >= observed).sum())
        p = hits / total
    else:
        signs = rng.choice((1.0, -1.0), size=(100_000, n))
        p = float((np.abs(signs @ d) / n >= observed).mean())
    return PairedDiff(float(d.mean()), float(lo), float(hi), p, n)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.maximum.accumulate((m - np.arange(m)) * p[ok][idx])
    out_ok = np.empty(m)
    out_ok[idx] = np.minimum(adj, 1.0)
    out[ok] = out_ok
    return out
