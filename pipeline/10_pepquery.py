"""PepQuery on a (cohort, peptide) row set, on every rank-1 full-length PSM of the peptide at spectrum_q <= 0.01.
usage: 10_pepquery.py <rows.tsv> <out_dir> [--entrapment] [--skip pq_by_cohort.tsv ...]; PEPQUERY_JAR names the jar.
--entrapment adds every ENTR target PSM of those cohorts; --skip leaves out tested rows. Writes pq_by_cohort.tsv."""
import os
import subprocess
import sys

import pandas as pd

from juncta.config import PDC2NAME
from juncta.match import rank1_psms
from juncta.peptide import REFERENCE
from juncta.spectra import mgf_blocks

rows_file, out, *options = sys.argv[1:]
os.makedirs(out, exist_ok=True)
rows = pd.read_csv(rows_file, sep="\t")
if "--skip" in options:
    done = pd.concat([pd.read_csv(f, sep="\t") for f in options[options.index("--skip") + 1:]])
    done = done[done.cls == "CHIM"]
    rows = rows[~(rows.cohort + "|" + rows.peptide).isin(set(done.cohort + "|" + done.peptide))]

queries, n_spectra = [], 0
with open(f"{out}/spectra.mgf", "w") as mgf:
    for pdc, name in PDC2NAME.items():
        if name in set(rows.cohort):
            psms = rank1_psms(f"results/cptac/pan/cohorts/{pdc}", ["filename", "scan", "is_decoy", "spectrum_q", "cls"])
            psms = psms[(psms.spectrum_q <= 0.01) & ~psms.is_decoy]
            psms = psms[((psms.cls == "CHIM") & psms.tpep.isin(set(rows[rows.cohort == name].peptide)))
                        | ((psms.cls == "ENTR") & ("--entrapment" in options))]
            titles = psms.filename.astype(str) + ":" + psms.scan.astype(str)
            wanted, ordinals = set(titles), {}
            for lines in mgf_blocks(f"results/cptac/pan/cohorts/{pdc}/cand_spectra.mgf.gz"):
                if lines[0][6:].strip() in wanted:
                    n_spectra += 1
                    ordinals[lines[0][6:].strip()] = n_spectra
                    # PepQuery reports spectrum_title as the 1-based ordinal of the spectrum.
                    mgf.write(f"BEGIN IONS\nTITLE=S{n_spectra}\n" + "".join(lines[1:]) + "END IONS\n")
            queries.append(psms[["stripped_peptide", "cls"]].assign(cohort=name, ordinal=titles.map(ordinals)))
queries = pd.concat(queries, ignore_index=True)
open(f"{out}/peptides.txt", "w").write("\n".join(sorted(set(queries.stripped_peptide))) + "\n")
subprocess.run(["java", "-Xmx120G", "-jar", os.environ["PEPQUERY_JAR"], "-i", f"{out}/peptides.txt", "-t", "peptide",
                "-ms", f"{out}/spectra.mgf", "-p", "TMT10_11", "-db", REFERENCE, "-o", f"{out}/pq", "-cpu", "24",
                "-fast", "-aa"], check=True, stdout=open(f"{out}/pepquery.log", "w"), stderr=subprocess.STDOUT)

ranks = pd.read_csv(f"{out}/pq/psm_rank.txt", sep="\t")
ranks = ranks.merge(queries, left_on=["spectrum_title", "peptide"], right_on=["ordinal", "stripped_peptide"])
ranks["ok"] = ranks.confident.astype(str).eq("Yes")
group = ["cohort", "stripped_peptide", "cls"]
by_cohort = queries.groupby(group).size().rename("n_spectra_tested").to_frame()
by_cohort = by_cohort.join(ranks.groupby(group).agg(n_confident=("ok", "sum"), best_p=("pvalue", "min")))
by_cohort = by_cohort.fillna({"n_confident": 0}).assign(pq_validated=lambda t: t.n_confident > 0).reset_index()
by_cohort.rename(columns={"stripped_peptide": "peptide"}).to_csv(f"{out}/pq_by_cohort.tsv", sep="\t", index=False)
