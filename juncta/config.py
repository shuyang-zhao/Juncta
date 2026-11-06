"""Cohorts and constants shared across the repository. Every script runs from a working directory holding data/ and
results/, so paths are relative to it; small curated tables tracked in git live in inputs/, read through INPUTS."""
from pathlib import Path

import numpy as np

INPUTS = Path(__file__).resolve().parents[1] / "inputs"

# PDC study -> (cohort, matched TCGA project, TMT label)
COHORTS = {
    "PDC000127": ("CCRCC", "KIRC", "tmt10"),
    "PDC000153": ("LUAD", "LUAD", "tmt10"),
    "PDC000125": ("UCEC", "UCEC", "tmt10"),
    "PDC000204": ("GBM", "GBM", "tmt11"),
    "PDC000116": ("COAD", "COAD", "tmt10"),
    "PDC000234": ("LSCC", "LUSC", "tmt11"),
    "PDC000221": ("HNSCC", "HNSC", "tmt11"),
    "PDC000270": ("PDA", "PAAD", "tmt11"),
    "PDC000120": ("BRCA", "BRCA", "tmt10"),
    "PDC000110": ("OV", "OV", "tmt10"),
}
PDC2NAME = {pdc: v[0] for pdc, v in COHORTS.items()}
NAME2PDC = {v[0]: pdc for pdc, v in COHORTS.items()}

LENGTH_BINS = [0, 8, 10, 12, 15, 19, 25, 35, 100]   # peptide-length strata of the decoupled null


def length_bin(length):
    return np.digitize(length, LENGTH_BINS[1:-1], right=True)
