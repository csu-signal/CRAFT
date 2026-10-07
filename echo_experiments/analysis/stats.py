"""
Uncertainty and paired comparisons over structure-level units.

The unit is the structure: every condition plays the same structures, and a
structure's episodes are averaged first (they share the structure, so they
aren't independent samples). With 20 structures, n = 20.

  * sem():         mean +- standard error of the mean (SD / sqrt(n), ddof=1)
                   over structure means. +-1 SEM is roughly a 68% interval,
                   not 95% -- figures and tables say "+- SEM".
  * paired_diff(): difference vs a reference on the structures both ran:
                   mean +- SEM of the per-structure differences, and a
                   sign-flip permutation p-value (exact up to 20 structures,
                   Monte Carlo beyond).
  * holm():        family-wise correction across the metrics compared.
"""
from dataclasses import dataclass

import numpy as np

ALPHA = 0.05


@dataclass
class Estimate:
    mean: float
    sem: float
    n: int

    @property
    def lo(self):
        return self.mean - self.sem

    @property
    def hi(self):
        return self.mean + self.sem


@dataclass
class PairedDiff:
    diff: float
    sem: float
    p: float
    n: int
    p_holm: float = float("nan")


def _sem(x):
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def sem(values):
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return Estimate(np.nan, np.nan, 0)
    return Estimate(float(x.mean()), _sem(x), len(x))


def paired_diff(a, b, seed=0):
    """a, b: pd.Series indexed by structure_idx. Returns a - b."""
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).to_numpy(dtype=float)
    n = len(d)
    if n < 2:
        return PairedDiff(float(d.mean()) if n else np.nan, np.nan, np.nan, n)

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
        signs = np.random.default_rng(seed).choice((1.0, -1.0), size=(100_000, n))
        p = float((np.abs(signs @ d) / n >= observed).mean())
    return PairedDiff(float(d.mean()), _sem(d), p, n)


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
