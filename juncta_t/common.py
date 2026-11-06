"""Constants and peptide helpers shared by the Juncta-T scripts (no torch dependency)."""
import re
import zlib
from pathlib import Path

import numpy as np
import yaml

from juncta.config import LENGTH_BINS
from juncta.peptide import AA

CONFIG = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
OB = CONFIG["optionb"]
RESULTS = "results/cptac/dl"
OPTIONB_COHORTS = ["PDC000127", "PDC000125", "PDC000204"]   # CCRCC, UCEC, GBM: the cohorts with local MGFs
N_STRATA = len(LENGTH_BINS) - 1
# set: folds, PSM evaluation, ET evaluation, ET score prefix, unit kind
SETS = {"dev_f0123": ((0, 1, 2, 3), "dev_f0123_a9", "eval_a9", "score", "full"),
        "test_f45": ((4, 5), "test_f45_a9", "test_a9", "test_score", "full"),
        "dev_f0123_trunc": ((0, 1, 2, 3), "dev_f0123_trunc_a9", "devtrunc_a9", "trunc_score", "full_trunc"),
        "test_f45_trunc": ((4, 5), "test_f45_trunc_a9", "test_trunc_a9", "trunc_score", "full_trunc")}
# Sage features stored per candidate; 10 = LDA score, 16 = charge, 19 = rank
FEATS = ["hyperscore", "delta_next", "delta_best", "matched_peaks", "longest_b", "longest_y", "longest_y_pct",
         "matched_intensity_pct", "log_scored_candidates", "poisson", "sage_discriminant_score", "fragment_ppm",
         "abs_precursor_ppm", "isotope_error", "missed_cleavages", "peptide_len", "charge", "abs_delta_rt_model",
         "log_ms2_intensity", "rank"]
PROTON, H2O, TMT, OX = 1.007276467, 18.010564684, 229.162932, 15.994915
SEARCH_MASS = {**AA, "K": AA["K"] + TMT}   # the C of juncta.peptide includes carbamidomethyl
# token ids: 0 pad, 1 candidate start, then residues; 'm' = oxidised Met, TMT and carbamidomethyl-Cys are static
TOKENS = ["<pad>", "<cand>"] + list("ACDEFGHIKLMNPQRSTVWY") + ["m"]


def crc(text):
    return zlib.crc32(text.encode())


def il(peptide):
    return peptide.replace("I", "L")


def sage_decoy(peptide):
    """Sage's decoy (tag rev_): first and last residue stay, the interior is reversed."""
    return peptide if len(peptide) <= 2 else peptide[0] + peptide[1:-1][::-1] + peptide[-1]


def plex_half(filename):
    """Plex half (0 or 1) of an MGF file: fractions of one plex stay together."""
    return crc(re.sub(r"(?:_f\d+[A-Za-z]?)?\.mgf$", "", filename)) % 2


def stratum_map(strata, min_n):
    """Strata with fewer than min_n pool members borrow the model of the nearest large stratum (ties: shorter)."""
    counts = np.bincount(np.asarray(strata, int), minlength=N_STRATA)[:N_STRATA]
    big = np.flatnonzero(counts >= min_n)
    mapping = np.arange(N_STRATA)
    for b in np.flatnonzero(counts < min_n):
        mapping[b] = big[np.argmin(np.abs(big - b) * 2 + (big > b))]
    return mapping
