"""Per-cohort decoupled null. usage: 08_cohort_stats.py <cohort_dir> <db_dir> <tcga_project|NONE> <out_dir>
Hypothesis = CHIMPEP entry, T = its best rank-1 full-length score, p = pi_hit(L) * S(T | L), with S the survival over
all decoy peptides. Writes global_tier.tsv, gpd_validation.tsv, pvals.tsv.gz, pvals_gpd.tsv.gz (p_gpd, its class BH)."""
import os
import sys

import numpy as np
import pandas as pd

from juncta.config import length_bin
from juncta.match import rank1_psms
from juncta.stats import bh, hit_rate, models_with_validation, surv

cdir, ddir, project, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
flags = pd.read_csv(f"{ddir}/chimpep_flags.tsv", sep="\t").set_index("entry")
# excl: I/L-equivalent to a reference peptide; bench: PUB/POS only; disc: everything else.
family = pd.Series(np.where(flags.ref_IL, "excl", np.where(flags.bench_only, "bench", "disc")), index=flags.index)
psms = rank1_psms(cdir, ["filename", "is_decoy", "spectrum_q", "sage_discriminant_score", "cls"])
psms["plex"] = psms.filename.str.replace(r"_f\w+\.mgf$", "", regex=True)
psms["family"] = psms.tpep.map(family).fillna("disc")   # ENTR headers hold the target, so ENTR inherit its family

rows = []
for name in ["disc", "bench"]:
    for q in [0.001, 0.01, 0.05]:
        row = {"family": name, "q": q}
        for (cls, decoy), g in psms[(psms.spectrum_q <= q) & (psms.family == name)].groupby(["cls", "is_decoy"]):
            row[f"{cls}_{'D' if decoy else 'T'}_pep"] = g.stripped_peptide.nunique()
            row[f"{cls}_{'D' if decoy else 'T'}_psm"] = len(g)
        rows.append(row)
pd.DataFrame(rows).fillna(0).to_csv(f"{out}/global_tier.tsv", sep="\t", index=False)

best = psms.groupby(["cls", "is_decoy", "stripped_peptide"]).agg(
    T=("sage_discriminant_score", "max"), q=("spectrum_q", "min"), n_psm=("spectrum_q", "size"),
    n_plex=("plex", "nunique"), tpep=("tpep", "first")).reset_index()
best = best[~best.is_decoy | (best.cls == "CHIM")]   # ENTR decoys are not hypotheses
best.index = np.where(best.is_decoy, "CHIMd|" + best.tpep, best.cls + "|" + best.stripped_peptide)
kept = pd.Series(open(f"{ddir}/chim_peptides_kept.txt").read().split())
entrapment = pd.read_csv(f"{ddir}/entrapment.tsv", sep="\t")
entries = pd.concat([pd.DataFrame({"cls": "CHIM", "decoy": False, "entry": kept, "target_of": kept}),
                     pd.DataFrame({"cls": "CHIM", "decoy": True, "entry": kept, "target_of": kept}),
                     pd.DataFrame({"cls": "ENTR", "decoy": False, "entry": entrapment.entrapment,
                                   "target_of": entrapment.target})], ignore_index=True)
key = np.where(entries.decoy, "CHIMd|" + entries.entry, entries.cls + "|" + entries.entry)
entries[["T", "q", "n_psm", "n_plex"]] = best[["T", "q", "n_psm", "n_plex"]].astype(float).reindex(key).values
entries["L"] = entries.entry.str.len()
entries["lb"] = length_bin(entries.L)
hit = hit_rate(entries["T"][entries.decoy].notna(), entries.lb[entries.decoy])   # pi_hit, from the CHIM decoy entries
decoys = pd.read_parquet(f"{cdir}/decoy_best.parquet", columns=["stripped_peptide", "T"])
models, F, validation = models_with_validation(decoys["T"].values, length_bin(decoys.stripped_peptide.str.len()), 1000)
validation.to_csv(f"{out}/gpd_validation.tsv", sep="\t", index=False)
p = np.ones(len(entries))
for b, (scores, n, *_) in models.items():
    sel = (entries.lb.values == b) & entries["T"].notna().values
    p[sel] = hit.get(b, hit.mean()) * (n - np.searchsorted(scores, entries["T"].values[sel], side="left") + 1) / (n + 1)
entries["p"] = p

groups = pd.read_csv(f"{ddir}/peptide_groups.tsv", sep="\t", low_memory=False)
groups = groups[groups.peptide.isin(set(kept))]
groups["matched"] = (groups.source == f"R3_{project}") & (pd.to_numeric(groups.r3_frac_T, errors="coerce") >= 0.05)
groups["pr"] = groups.name_d.fillna("?").astype(str) + "-" + groups.name_a.fillna("?").astype(str)
annotation = groups.groupby("peptide").agg(sources=("source", lambda s: ",".join(sorted(set(s)))),
                                           pair=("pr", lambda s: ",".join(sorted(set(s)))[:200]),
                                           orf_class=("orf_class", lambda s: ",".join(sorted(set(s.dropna())))),
                                           matched=("matched", "any"))
entries = entries.join(annotation, on="target_of")
entries["family"] = entries.target_of.map(family).fillna("disc")
bench = (entries.cls == "CHIM") & ~entries.decoy & (entries.family == "bench")
entries.loc[bench, "bench_bh0.05"] = bh(entries.p[bench], 0.05)
entries.to_csv(f"{out}/pvals.tsv.gz", sep="\t", index=False)

# The tail step works on the written table, so that both p-value tables hold the same parsed scores.
entries = pd.read_csv(f"{out}/pvals.tsv.gz", sep="\t", low_memory=False)
p = np.ones(len(entries))
for b, model in models.items():
    sel = (entries.lb.values == b) & entries["T"].notna().values
    p[sel] = hit.get(b, hit.mean()) * surv(entries["T"].values[sel], model, F)
entries["p_gpd"] = np.minimum(p, 1.0)
for decoy in [False, True]:
    m = (entries.cls == "CHIM") & (entries.decoy == decoy) & (entries.family == "disc")
    for alpha in [0.01, 0.05]:
        entries.loc[m, f"bhg{alpha}"] = bh(entries.p_gpd[m], alpha)
entries.to_csv(f"{out}/pvals_gpd.tsv.gz", sep="\t", index=False)
