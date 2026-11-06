"""Empirical survival up to the K-th largest decoy and a GPD fit beyond it (Knijnenburg et al. 2009, Bioinformatics
25:i161), inflated by the worst split-half observed / predicted ratio; and Benjamini-Hochberg."""
import numpy as np
import pandas as pd
from scipy.stats import genpareto


def fit(s, K):
    """GPD fit to the top K values of s. Returns (sorted s, n, k, threshold, shape, scale)."""
    s = np.sort(s)
    n = len(s)
    k = min(K, max(50, n // 20))
    u = s[n - k - 1]
    y = s[n - k:] - u
    c, _, sc = genpareto.fit(y, floc=0)
    if c < 0:   # a bounded fit would give zero survival beyond its endpoint
        c, sc = 0.0, float(y.mean())
    return s, n, k, u, c, sc


def surv(T, m, f=1.0):
    """Survival of scores T under model m; the GPD part is inflated by f and never exceeds the empirical value."""
    s, n, k, u, c, sc = m
    T = np.asarray(T, float)
    out = (n - np.searchsorted(s, T, side="left") + 1) / (n + 1)
    hi = T > u
    out[hi] = np.minimum(out[hi], f * (k / (n + 1)) * genpareto.sf(T[hi] - u, c, loc=0, scale=sc))
    return out


def models_with_validation(values, strata, K, seed=7):
    """Fit one tail per stratum and check it on held-out halves. Returns ({stratum: model}, inflation factor F,
    validation table). One random generator walks the strata in sorted order, so that order is part of the result."""
    rng = np.random.default_rng(seed)
    val, models = [], {}
    for b, g in pd.DataFrame({"x": values, "lb": strata}).groupby("lb"):
        x = g["x"].values
        models[b] = fit(x, K)
        idx = rng.permutation(len(x))
        A, B = x[idx[: len(x) // 2]], x[idx[len(x) // 2:]]
        mA = fit(A, K)
        sB = np.sort(B)
        for q in [1e-3, 1e-4, 3e-5, 1e-5]:
            lo, hi = mA[3], mA[0][-1] + 50   # bisection for the threshold where the half-A fit predicts survival q
            for _ in range(80):
                mid = (lo + hi) / 2
                (lo, hi) = (mid, hi) if surv([mid], mA)[0] > q else (lo, mid)
            n_above = len(sB) - np.searchsorted(sB, (lo + hi) / 2, side="left")
            val.append({"lb": b, "pred": q, "n_obs_B": int(n_above), "expected_n_B": q * len(sB)})
    V = pd.DataFrame(val)
    agg = V[V.pred <= 3e-5].groupby("pred").agg(e=("expected_n_B", "sum"), o=("n_obs_B", "sum"))
    return models, float(max(1.0, ((agg.o + 1) / (agg.e + 1)).max())), V


def apply_surv(values, strata, models, F):
    out = np.ones(len(values))
    for b, m in models.items():
        sel = strata == b
        if sel.any():
            out[sel] = surv(values[sel], m, F)
    return out


def bh(p, alpha):
    """Boolean rejections of the Benjamini-Hochberg procedure at level alpha."""
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    ok = p[order] <= alpha * np.arange(1, m + 1) / m
    k = np.max(np.nonzero(ok)[0]) + 1 if ok.any() else 0
    rejected = np.zeros(m, bool)
    rejected[order[:k]] = True
    return rejected


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    adjusted[order] = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    return np.minimum(adjusted, 1)
