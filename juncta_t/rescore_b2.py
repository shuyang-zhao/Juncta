"""B2, the MS2Rescore-style baseline: Sage + MS2PIP-TMT + DeepLC features, fitted with mokapot's Percolator (3-fold
over spectra, trained on the Sage rank-1 candidates) and applied to every top-5 candidate of an eval shard.
usage: python -m juncta_t.rescore_b2 <PDC> <fold>     -> b2/v3_<PDC>_f<fold>/scores.npy (n x K, -inf for empty slots)"""
import os
import sys

import deeplc
import mokapot
import numpy as np
import pandas as pd
import torch
from ms2pip.core import predict_batch
from psm_utils import PSM, PSMList

from juncta_t.common import RESULTS, crc

PROCS = 4
N_MS2PIP, N_DEEPLC = 9, 2


def unimod(peptide):
    return (peptide.replace("[+229.16293]-", "[TMT6plex]-").replace("K[+229.16293]", "K[TMT6plex]")
            .replace("C[+57.021465]", "C[Carbamidomethyl]").replace("M[+15.994915]", "M[Oxidation]"))


def correlation(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) >= 3 and a.std() != 0 and b.std() != 0 else 0.0


def ms2_features(theoretical_mz, predicted_log2, is_b, mz, it, tol=20e-6):
    """Pearson and spearman of log2 intensities, cosine, pearson of b and y ions, top-6 matched, explained, MSE."""
    observed = it.astype(np.float64) ** 2   # the shard stores sqrt(I / max)
    mz, observed = mz[mz > 0], observed[mz > 0]
    if len(theoretical_mz) == 0 or len(mz) == 0:
        return np.zeros(N_MS2PIP)
    position = np.searchsorted(mz, theoretical_mz)
    best = np.zeros(len(theoretical_mz))
    for offset in (-1, 0):
        j = np.clip(position + offset, 0, len(mz) - 1)
        best = np.where(np.abs(mz[j] - theoretical_mz) <= tol * theoretical_mz, np.maximum(best, observed[j]), best)
    observed_log2 = np.log2(best + 0.001)
    predicted = np.clip(2.0 ** predicted_log2 - 0.001, 0, None)
    norm = np.linalg.norm(best) * np.linalg.norm(predicted)
    return np.array([correlation(observed_log2, predicted_log2),
                     correlation(np.argsort(np.argsort(observed_log2)).astype(float),
                                 np.argsort(np.argsort(predicted_log2)).astype(float)),
                     float(best @ predicted / norm) if norm > 0 else 0.0,
                     correlation(observed_log2[is_b], predicted_log2[is_b]),
                     correlation(observed_log2[~is_b], predicted_log2[~is_b]),
                     float((best[np.argsort(-predicted_log2)[:6]] > 0).mean()),
                     float(predicted[best > 0].sum() / max(predicted.sum(), 1e-12)),
                     float(best.sum() / max(observed.sum(), 1e-12)),
                     float(np.mean((observed_log2 - predicted_log2) ** 2))])


def ms2pip_features(cfeat, mz, it, ii, kk, peptides):
    """MS2PIP 4.2 model "TMT", predicted once per peptidoform and charge."""
    charge = np.clip(cfeat[ii, kk, 16].astype(int), 1, 6)
    keys = pd.Series(peptides).astype(str) + "/" + pd.Series(charge).astype(str)
    unique_keys, inverse = np.unique(keys.values, return_inverse=True)
    X = np.zeros((len(ii), N_MS2PIP), np.float32)
    order = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[order], np.arange(len(unique_keys) + 1))
    for c0 in range(0, len(unique_keys), 200_000):
        chunk = unique_keys[c0:c0 + 200_000]
        psms = PSMList(psm_list=[PSM(peptidoform=x, spectrum_id=str(c0 + j)) for j, x in enumerate(chunk)])
        for j, result in enumerate(predict_batch(psms, model="TMT", processes=PROCS)):
            b_mz, y_mz = result.theoretical_mz["b"], result.theoretical_mz["y"]
            theoretical = np.concatenate([b_mz, y_mz]).astype(np.float64)
            predicted = result.predicted_intensity
            predicted = np.concatenate([predicted["b"], predicted["y"]]).astype(np.float64)
            is_b = np.r_[np.ones(len(b_mz), bool), np.zeros(len(y_mz), bool)]
            for row in order[starts[c0 + j]:starts[c0 + j + 1]]:
                X[row] = ms2_features(theoretical, predicted, is_b, np.asarray(mz[ii[row]]), np.asarray(it[ii[row]]))
    return X


def deeplc_features(meta, prec, label, ii, peptides, K):
    """|dRT| and dRT^2 from DeepLC 4.5, calibrated on <= 20,000 confidently labelled reference PSMs of the shard."""
    unique_peptides, peptide_index = np.unique(peptides, return_inverse=True)
    confident = np.flatnonzero((label >= 0) & (label < K) & (meta.orig_q.fillna(1).values <= 0.001))
    confident = np.random.default_rng(20261103).permutation(confident)[:20000]
    reference = PSMList(psm_list=[PSM(peptidoform=unimod(meta.cand_peptide.values[i][label[i]]), spectrum_id=str(i),
                                      retention_time=float(prec[i, 2])) for i in confident])
    rt = deeplc.predict_and_calibrate([unimod(x) for x in unique_peptides], psm_list_reference=reference)
    delta = prec[ii, 2] - np.asarray(rt, float)[peptide_index]
    return np.c_[np.abs(delta), delta ** 2]


def fit_and_score(X, cfeat, decoy, ii, kk, peptides, n, K):
    """Every PSM is scored by a model that never saw its spectrum."""
    F = np.c_[cfeat[ii, kk, :], X].astype(np.float64)
    F[~np.isfinite(F)] = 0.0
    names = [f"feature{j}" for j in range(F.shape[1])]
    table = pd.DataFrame(F, columns=names).assign(target=~decoy[ii, kk], spec=ii, pep=peptides,
                                                  rank1=cfeat[ii, kk, 19] == 1)
    fold = np.array([crc(str(i)) % 3 for i in range(n)])[ii]
    scores = np.full((n, K), -np.inf)
    columns = dict(target_column="target", spectrum_columns=["spec"], peptide_column="pep", feature_columns=names)
    for f in range(3):
        training = table[(fold != f) & table.rank1]
        training = training.sample(300_000, random_state=f)
        model = mokapot.PercolatorModel(train_fdr=0.01, max_iter=10, n_jobs=PROCS, rng=f)
        model.fit(mokapot.LinearPsmDataset(training, **columns))
        scores[ii[fold == f], kk[fold == f]] = model.predict(mokapot.LinearPsmDataset(table[fold == f], **columns))
    return scores


if __name__ == "__main__":
    torch.set_num_threads(PROCS)
    shard = f"{RESULTS}/shards_v3/{sys.argv[1]}_f{sys.argv[2]}_eval"
    meta = pd.read_parquet(f"{shard}/meta.parquet", columns=["cand_peptide", "orig_q"])
    cm, decoys, cfeats, precursors = (np.load(f"{shard}/{name}.npy") for name in ("cmask", "cdecoy", "cfeat", "prec"))
    labels = np.load(f"{shard}/label.npy").astype(int)
    rows, slots = np.nonzero(cm)
    candidates = np.array([meta.cand_peptide.values[i][k] for i, k in zip(rows, slots)], dtype=object)
    features = np.zeros((len(rows), N_MS2PIP + N_DEEPLC), np.float32)
    features[:, :N_MS2PIP] = ms2pip_features(cfeats, np.load(f"{shard}/mz.npy", mmap_mode="r"),
                                             np.load(f"{shard}/it.npy", mmap_mode="r"), rows, slots, candidates)
    features[:, N_MS2PIP:] = deeplc_features(meta, precursors, labels, rows, candidates, cm.shape[1])
    out = f"{RESULTS}/b2/v3_{sys.argv[1]}_f{sys.argv[2]}"
    os.makedirs(out, exist_ok=True)
    np.save(f"{out}/scores.npy", fit_and_score(features, cfeats, decoys, rows, slots, candidates, *cm.shape))
