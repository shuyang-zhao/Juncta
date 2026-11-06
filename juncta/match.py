"""Full-length matching of Sage PSMs, which list every entry containing the peptide: CHIMPEP targets match on the header
sequence, CHIMPEP decoys on length, preferring the exact reversal; ENTR headers hold the target and match on length."""
import numpy as np
import pyarrow.parquet as pq


def annotate(X):
    """Add tpep (the entry or target peptide of each PSM) and full (a full-length match) to a PSM table, in place."""
    tpep = []
    for accessions, s in zip(X.proteins.values, X.stripped_peptide.values):
        best = None
        for accession in str(accessions).split(";"):
            if "|" not in accession:
                continue
            name, pep = accession.split("|", 1)
            decoy, chim = name.startswith("rev_"), "CHIMPEP_" in name
            if chim and not decoy:
                if pep == s:
                    best = pep
                    break
            elif len(pep) == len(s):
                if decoy and chim and pep[:-1][::-1] + pep[-1:] == s:
                    best = pep
                    break
                if best is None:
                    best = pep
        tpep.append(best)
    X["tpep"] = np.array(tpep, dtype=object)
    X["full"] = np.array([t is not None for t in tpep], dtype=bool)
    return X


def rank1_psms(cohort_dir, columns, classes=("CHIM", "ENTR")):
    """Rank-1 full-length PSMs of class_cands.parquet; columns come besides proteins and stripped_peptide."""
    psms = pq.read_table(f"{cohort_dir}/class_cands.parquet", columns=["stripped_peptide", "proteins", *columns],
                         filters=[("rank", "=", 1), ("cls", "in", classes)]).to_pandas()
    psms = annotate(psms)
    return psms[psms.full].copy()
