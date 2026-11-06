"""Steps 08 to 11 as scripts on small synthetic screens, in a scratch working directory: a runnable example of the files
each step reads. A stub on PATH stands in for the java call of PepQuery."""
import gzip
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from juncta.config import COHORTS, NAME2PDC
from juncta.peptide import ions, parse_modified, shuffle_entrapments

pytest.importorskip("ahocorasick")
REPO = Path(__file__).resolve().parents[1]
PAN = "results/cptac/pan"
PSM_COLUMNS = ["stripped_peptide", "peptide", "proteins", "rank", "cls", "filename", "scan", "charge", "is_decoy",
               "spectrum_q", "sage_discriminant_score"]
PSM_TYPES = {"stripped_peptide": str, "peptide": str, "proteins": str, "cls": str, "filename": str, "is_decoy": bool,
             "rank": int, "scan": int, "charge": int, "spectrum_q": float, "sage_discriminant_score": float}
FILE = "plexA_f01.mgf"
OXIDISED = "[+229.16293]-AMAAAM[+15.994915]AK[+229.16293]"
STRONG, TESTED_OUT, NO_ELIGIBLE_PSM, WEAK = "AMAAAMAK", "GLGSWDVK", "WWGGTTYR", "DFGNSLHK"
STUB = """#!/usr/bin/env python3
import os, sys
arg = lambda flag: sys.argv[sys.argv.index(flag) + 1]
peptides = open(arg("-i")).read().split()
n_spectra = open(arg("-ms")).read().count("BEGIN IONS")
os.makedirs(arg("-o"), exist_ok=True)
with open(arg("-o") + "/psm_rank.txt", "w") as out:
    out.write("spectrum_title\\tpeptide\\tconfident\\tpvalue\\n")
    for i in range(1, n_spectra + 1):
        for p in peptides:
            out.write(f"{i}\\t{p}\\t{'Yes' if str(i) in os.environ['PQ_CONFIDENT'].split(',') else 'No'}\\t0.001\\n")
"""


def filler(n):
    rng = np.random.default_rng(0)
    return sorted({"".join(rng.choice(list("ADEFGHLNQSTVY"), 7)) + "K" for _ in range(n)})


def write_screen(root, psms, peptides, decoy_hits):
    """psms: {cohort: [(peptide, modified, score, spectrum_q), ...]}; decoy_hits: CHIM decoys with a hit per cohort."""
    (root / "data/cptac_ref").mkdir(parents=True)
    (root / "data/cptac_ref/ref_pq.fasta").write_text(">ref\nAMAGGGGGGMAK\n")   # AMA and MAK are reference-shared ends
    db = root / PAN / "db_v1"
    db.mkdir(parents=True)
    keep = sorted(peptides)
    (db / "chim_peptides_kept.txt").write_text("\n".join(keep) + "\n")
    pairs = pd.DataFrame(shuffle_entrapments(keep), columns=["entrapment", "target"])
    pairs.to_csv(db / "entrapment.tsv", sep="\t", index=False)
    pd.DataFrame({"entry": keep, "sources": "DRS", "ref_IL": False, "bench_only": False}).to_csv(
        db / "chimpep_flags.tsv", sep="\t", index=False)
    pd.DataFrame({"peptide": keep, "source": "DRS", "r3_frac_T": np.nan, "name_d": "G1", "name_a": "G2",
                  "orf_class": "in_frame"}).to_csv(db / "peptide_groups.tsv", sep="\t", index=False)
    rng = np.random.default_rng(1)
    pool = filler(4000)
    for pdc, (name, *_) in COHORTS.items():
        cohort = root / PAN / "cohorts" / pdc
        cohort.mkdir(parents=True)
        pd.DataFrame({"stripped_peptide": pool, "T": rng.exponential(size=len(pool))}).to_parquet(
            cohort / "decoy_best.parquet")
        rows = [(pep[-2::-1] + pep[-1], pep[-2::-1] + pep[-1], f"rev_CHIMPEP_{i}|{pep}", 1, "CHIM", FILE, 1000 + i, 2,
                 True, 0.9, 0.05) for i, pep in enumerate(keep[:decoy_hits])]
        with gzip.open(cohort / "cand_spectra.mgf.gz", "wt") as mgf:
            for scan, (pep, modified, score, q) in enumerate(psms.get(name, []), start=1):
                accession = f"CHIMPEP_{keep.index(pep)}|{pep}"
                rows.append((pep, modified, accession, 1, "CHIM", FILE, scan, 2, False, q, score))
                peaks = ions(*parse_modified(modified))[0][3:5]
                mgf.write(f"BEGIN IONS\nTITLE={FILE}:{scan}\nPEPMASS=500.0\nCHARGE=2+\n"
                          + "".join(f"{mz:.5f} 100.0\n" for mz in peaks) + "END IONS\n")
        pd.DataFrame(rows, columns=PSM_COLUMNS).astype(PSM_TYPES).to_parquet(cohort / "class_cands.parquet")


def run(root, *script, **environment):
    env = {**os.environ, "PYTHONPATH": str(REPO), "OMP_NUM_THREADS": "1", **environment}
    return subprocess.Popen([sys.executable, str(REPO / "pipeline" / script[0]), *script[1:]], cwd=root, env=env)


def statistics(root):
    jobs = [run(root, "08_cohort_stats.py", f"{PAN}/cohorts/{pdc}", f"{PAN}/db_v1", "NONE", f"{PAN}/analysis/{pdc}")
            for pdc in COHORTS]
    assert all(job.wait() == 0 for job in jobs)
    assert run(root, "09_pan.py").wait() == 0


def read(root, path):
    return pd.read_csv(root / PAN / path, sep="\t")


def test_a_screen_without_discoveries_runs_to_an_empty_table(tmp_path):
    """No CHIM decoy hit at all, and a weak target: its p must not vanish, and no step may fail on the empty lists."""
    peptides = [WEAK, *filler(2)]
    write_screen(tmp_path, {"CCRCC": [(WEAK, WEAK, 0.1, 0.99)]}, peptides, decoy_hits=0)
    statistics(tmp_path)
    entry = read(tmp_path, "tippett/entries_agg.tsv.gz").query("entry == @WEAK and cls == 'CHIM' and not decoy").iloc[0]
    assert entry[f"p_{NAME2PDC['CCRCC']}"] > 0.1 and entry.pagg_tippett > 0.1 and not entry.bh01_pagg_tippett
    assert read(tmp_path, "tippett/summary.tsv").discoveries.tolist() == [0, 0]
    assert read(tmp_path, "candidate_rows.tsv").empty
    # PepQuery has nothing to test: no jar, no java.
    assert run(tmp_path, "10_pepquery.py", f"{PAN}/candidate_rows.tsv", f"{PAN}/pepquery_hits").wait() == 0
    assert run(tmp_path, "11_main_table.py").wait() == 0
    assert read(tmp_path, "main_hits.tsv").empty


def test_pepquery_covers_every_candidate_and_tells_untested_from_failed(tmp_path):
    psms = {"CCRCC": [(STRONG, OXIDISED, 40.0, 0.0005)], "UCEC": [(STRONG, OXIDISED, 0.5, 0.005)],
            "GBM": [(TESTED_OUT, TESTED_OUT, 40.0, 0.0005)], "LUAD": [(NO_ELIGIBLE_PSM, NO_ELIGIBLE_PSM, 40.0, 0.5)]}
    peptides = [STRONG, TESTED_OUT, NO_ELIGIBLE_PSM, *filler(60)]
    write_screen(tmp_path, psms, peptides, decoy_hits=30)
    statistics(tmp_path)
    rows = set(read(tmp_path, "candidate_rows.tsv").itertuples(index=False, name=None))
    assert {("CCRCC", STRONG), ("UCEC", STRONG)} <= rows   # UCEC holds no class-FDR hit of it, yet it is tested
    assert ("UCEC", STRONG) not in set(read(tmp_path, "hits_class5_gpd.tsv").itertuples(index=False, name=None))

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "java").write_text(STUB)
    (bin_dir / "java").chmod(0o755)
    # Spectra follow the cohort order of the config: CCRCC (1), UCEC (2), GBM (3). Only the UCEC one is confident.
    environment = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "PEPQUERY_JAR": "stub.jar", "PQ_CONFIDENT": "2"}
    pepquery = run(tmp_path, "10_pepquery.py", f"{PAN}/candidate_rows.tsv", f"{PAN}/pepquery_hits", **environment)
    assert pepquery.wait() == 0
    assert run(tmp_path, "11_main_table.py").wait() == 0
    main = read(tmp_path, "main_hits.tsv").set_index("peptide")
    assert main.status[STRONG] == "main" and main.pq_status[STRONG] == "validated"
    assert main.pq_cohorts_tested[STRONG] == "CCRCC,UCEC" and main.pq_cohorts_confident[STRONG] == "UCEC"
    assert main.status[TESTED_OUT] == "not main" and main.pq_status[TESTED_OUT] == "failed"
    assert main.status[NO_ELIGIBLE_PSM] == "not main" and main.pq_status[NO_ELIGIBLE_PSM] == "not_tested"
