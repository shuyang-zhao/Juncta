"""Cross-cohort lists and the pan-cancer Tippett test: per cohort x = -ln S(T | L) (no pi_hit; 0 with no rank-1 hit),
A = the maximum x against the union pool of decoys, p_agg = pi_hit_agg(L) * S_A(A | L), one BH in the disc CHIM class.
Writes summary/global_tier_all.tsv, mainlist/mainlist_by_cohort.tsv, hits_class5_gpd.tsv, tippett/."""
import os

import numpy as np
import pandas as pd

from juncta.config import PDC2NAME, length_bin
from juncta.match import rank1_psms
from juncta.stats import apply_surv, bh, models_with_validation

PAN = "results/cptac/pan"
AGG = "results/cptac/pan/tippett"
for directory in (AGG, f"{PAN}/summary", f"{PAN}/mainlist", f"{PAN}/shortlist_gpd_a0.01", f"{PAN}/shortlist_gpd_a0.05"):
    os.makedirs(directory, exist_ok=True)
flags = pd.read_csv(f"{PAN}/db_v1/chimpep_flags.tsv", sep="\t").set_index("entry")
tiers, mainlist, class5, entries, pool = [], [], [], None, {}
for pdc, name in PDC2NAME.items():
    tiers.append(pd.read_csv(f"{PAN}/analysis/{pdc}/global_tier.tsv", sep="\t"))
    tiers[-1].insert(0, "cohort", name)
    # The global-1 % comparability list: CHIMPEP targets with a rank-1 full-length PSM at spectrum_q <= 0.01, except
    # entries I/L-equivalent to a reference peptide.
    psms = rank1_psms(f"{PAN}/cohorts/{pdc}", ["is_decoy", "spectrum_q"], ["CHIM"])
    psms = psms[(psms.spectrum_q <= 0.01) & ~psms.is_decoy]
    mainlist += [(name, p) for p in sorted(set(psms.tpep) - set(flags.index[flags.ref_IL]))]

    table = pd.read_csv(f"{PAN}/analysis/{pdc}/pvals_gpd.tsv.gz", sep="\t", low_memory=False,
                        usecols=["cls", "decoy", "entry", "target_of", "L", "lb", "family", "pair", "sources", "T",
                                 "p_gpd", "bhg0.05"])
    hits = table[(table.cls == "CHIM") & ~table.decoy & (table.family == "disc") & table["bhg0.05"].astype(bool)]
    class5 += [(name, p) for p in hits.entry]
    decoys = pd.read_parquet(f"{PAN}/cohorts/{pdc}/decoy_best.parquet", columns=["stripped_peptide", "T"])
    strata = length_bin(decoys.stripped_peptide.str.len())
    models, F, _ = models_with_validation(decoys["T"].values, strata, 1000)
    has_hit = table["T"].notna().values
    survival = np.ones(len(table))
    survival[has_hit] = apply_surv(table["T"].values[has_hit], table.lb.values[has_hit], models, F)
    if entries is None:   # every cohort has the same entries in the same order
        entries = table.drop(columns=["T", "p_gpd", "bhg0.05"])
    entries[f"x_{pdc}"] = np.where(has_hit, -np.log(survival), 0.0)
    entries[f"p_{pdc}"] = table.p_gpd.values
    x_pool = -np.log(apply_surv(decoys["T"].values, strata, models, F))
    pool[pdc] = pd.Series(x_pool, index=decoys.stripped_peptide.values)
pd.concat(tiers).to_csv(f"{PAN}/summary/global_tier_all.tsv", sep="\t", index=False)
for rows, path in ((mainlist, "mainlist/mainlist_by_cohort.tsv"), (class5, "hits_class5_gpd.tsv")):
    pd.DataFrame(rows, columns=["cohort", "peptide"]).to_csv(f"{PAN}/{path}", sep="\t", index=False)

# The row order of the union pool feeds the permutation of the tail validation.
union_pool = pd.concat(pool, axis=1).fillna(0.0)
strata = length_bin(pd.Series(union_pool.index).str.len().values)
entry_x = entries[[f"x_{pdc}" for pdc in PDC2NAME]].values
A = entry_x.max(axis=1)
chim_decoy = ((entries.cls == "CHIM") & entries.decoy).values
pi = pd.Series((entry_x > 0).any(axis=1)[chim_decoy]).groupby(entries.lb.values[chim_decoy]).mean()
models, F, _ = models_with_validation(union_pool.values.max(axis=1), strata, 1000)
survival = apply_surv(A, entries.lb.values, models, F)
entries["pagg_tippett"] = np.where(A > 0, np.minimum(1, entries.lb.map(pi).fillna(pi.mean()).values * survival), 1.0)

disc = (entries.family == "disc").values
target = (entries.cls == "CHIM").values & ~entries.decoy.values & disc
decoy = chim_decoy & disc
entrapment = (entries.cls == "ENTR").values & disc
p = entries.pagg_tippett.values
rows = []
for alpha, column in [(0.01, "bh01_pagg_tippett"), (0.05, "bh05_pagg_tippett")]:
    rejected = bh(p[target], alpha)
    cut = p[target][rejected].max()
    entries[column] = False
    entries.loc[target, column] = rejected
    rows.append(dict(alpha=alpha, union_pool=len(union_pool), discoveries=int(rejected.sum()), p_cutoff=cut,
                     chim_decoys_under_cutoff=int((p[decoy] <= cut).sum()),
                     chim_decoys_own_bh=int(bh(p[decoy], alpha).sum()),
                     entr_under_cutoff=int((p[entrapment] <= cut).sum())))
pd.DataFrame(rows).to_csv(f"{AGG}/summary.tsv", sep="\t", index=False)
entries.to_csv(f"{AGG}/entries_agg.tsv.gz", sep="\t", index=False)
