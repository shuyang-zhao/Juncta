"""Neighbour rule and the pan-cancer main table, after 09_pan.py and 10_pepquery.py.
Main hit = Tippett class FDR <= 1 % AND PepQuery confident in >= 1 cohort AND neighbour rule in >= 1 cohort.
The same checks at Tippett 5 % give the sensitivity tier. pq_status keeps a peptide without an eligible PSM (not_tested)
apart from one that PepQuery rejected (failed). Writes results/cptac/pan/main_hits.tsv."""
import numpy as np
import pandas as pd

from juncta.config import PDC2NAME
from juncta.match import rank1_psms
from juncta.peptide import ends, matched_ions, reference_pieces, specific_sites
from juncta.spectra import read_mgf_gz, top_peaks

PAN = "results/cptac/pan"
KEY = ["cohort", "peptide"]


def read_tsv(path):
    return pd.read_csv(path, sep="\t", low_memory=False)


def cohorts(rows, mask):
    """Per peptide, the cohorts of the rows under mask, in row order."""
    return rows[mask].groupby("peptide").cohort.agg(",".join)


agg = read_tsv("results/cptac/pan/tippett/entries_agg.tsv.gz")
agg = agg[(agg.cls == "CHIM") & ~agg.decoy & (agg.family == "disc")].set_index("entry")
tippett5 = agg[agg.bh05_pagg_tippett]
pvals = pd.concat([pd.read_csv(f"{PAN}/analysis/{pdc}/pvals_gpd.tsv.gz", sep="\t", low_memory=False,
                               usecols=["cls", "decoy", "entry", "bhg0.01", "bhg0.05"]).assign(cohort=name)
                   for pdc, name in PDC2NAME.items()])
pvals = pvals[(pvals.cls == "CHIM") & ~pvals.decoy].rename(columns={"entry": "peptide"})
rows = read_tsv(f"{PAN}/candidate_rows.tsv")

# Neighbour rule on the best PSM of every row.
pieces = reference_pieces(sorted(set(rows.peptide)))
rule_rows = []
for pdc, name in PDC2NAME.items():
    psms = rank1_psms(f"{PAN}/cohorts/{pdc}", ["peptide", "filename", "scan", "is_decoy", "charge", "spectrum_q",
                                               "sage_discriminant_score"], ["CHIM"])
    psms = psms.rename(columns={"peptide": "modified", "stripped_peptide": "peptide"})
    psms = psms[psms.peptide.isin(rows.peptide[rows.cohort == name]) & ~psms.is_decoy & (psms.tpep == psms.peptide)
                & (psms.spectrum_q <= 0.1)].copy()
    best = psms.sort_values("sage_discriminant_score").groupby("peptide").tail(1)
    best.index = best.filename.astype(str) + ":" + best.scan.astype(str)
    for title, _, _, _, mz, it in read_mgf_gz(f"{PAN}/cohorts/{pdc}/cand_spectra.mgf.gz", set(best.index)):
        p = best.peptide[title]
        hit_b, hit_y = matched_ions(best.modified[title], int(best.charge[title]), top_peaks(mz, it)[0])
        rule_rows.append((name, p, len(specific_sites(p, hit_b, hit_y, *ends(p, pieces))) >= 2))

pepquery = read_tsv(f"{PAN}/pepquery_hits/pq_by_cohort.tsv")
pepquery = pepquery[pepquery.cls == "CHIM"].drop_duplicates(KEY)
rows = rows.merge(pvals, on=KEY).merge(pepquery[[*KEY, "n_spectra_tested", "pq_validated"]], on=KEY, how="left")
rows = rows.merge(pd.DataFrame(rule_rows, columns=[*KEY, "rule_ok"]), on=KEY, how="left")
rows = rows.sort_values("cohort", kind="stable")
for column in ["bhg0.01", "bhg0.05", "pq_validated", "rule_ok"]:
    rows[column] = rows[column].eq(True)

main = agg.loc[sorted(set(rows.peptide)), ["pair", "pagg_tippett"]].rename(columns={"pagg_tippett": "tippett_p"})
main.index.name = "peptide"
main["tippett_tier"] = np.where(main.index.isin(agg.index[agg.bh01_pagg_tippett]), "1%",
                                np.where(main.index.isin(tippett5.index), "5%", ""))
main["cohorts_class1"], main["cohorts_class5"] = cohorts(rows, rows["bhg0.01"]), cohorts(rows, rows["bhg0.05"])
main["pq_cohorts_tested"] = cohorts(rows, rows.n_spectra_tested > 0)
main["pq_cohorts_confident"], main["rule_cohorts"] = cohorts(rows, rows.pq_validated), cohorts(rows, rows.rule_ok)
main["pq_status"] = np.where(main.pq_cohorts_confident.notna(), "validated",
                             np.where(main.pq_cohorts_tested.notna(), "failed", "not_tested"))
passed = (main.tippett_tier != "") & main.pq_cohorts_confident.notna() & main.rule_cohorts.notna()
main["status"] = np.where(~passed, "not main", np.where(main.tippett_tier == "1%", "main", "sensitivity"))
rank = {"main": 0, "sensitivity": 1, "not main": 2}
main = main.sort_values(["status", "tippett_p"], key=lambda s: s.map(rank) if s.name == "status" else s)
main.to_csv(f"{PAN}/main_hits.tsv", sep="\t")
