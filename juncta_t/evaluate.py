"""Class-FDR evaluation on the known truth of the option-B worlds, the same for every scorer; production calibration.
A world or unit gives x = -ln S(T | L) over all decoys; A = max x per fold; p = pi_hit(L) x S_A(A | L); BH in the class.
usage: python -m juncta_t.evaluate psm|R3_ET0|R1b_ETk <set of juncta_t.common.SETS> | prod R3_ET0|R1b_ETk | e2"""
import functools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binom, ks_2samp

from juncta.config import length_bin
from juncta.peptide import matched_ions, parse_modified, specific_sites
from juncta.stats import apply_surv, bh, hit_rate, models_with_validation, surv
from juncta_t.common import OPTIONB_COHORTS, RESULTS, SETS, crc, il, sage_decoy, stratum_map
from juncta_t.model_entry import TOKEN_FEATS, fold_entries, fold_worlds, load_unit, load_world, prod_entries

CLASS = ["PSEUDO", "REJ", "NNEG", "MOD"]
KNOWN_FALSE = ["REJ", "NNEG", "MOD"]
MIN_STRATUM = 200
CUTOFFS = (1e-2, 1e-3, 1e-4)
BOOT_SEED = 20261103
N_BOOT = 1000
ET_NAMES = {"R3_ET0": "ET0", "R1b_ETk": "ETk"}


def final_survival(values, strata, models, F):
    """Survival of A, never of the per-world x: empirical with >= 10 null exceedances, else min(empirical, F x GPD)."""
    out = np.ones(len(values))
    for b, model in models.items():
        T = np.asarray(values[strata == b], float)
        exceed = model[1] - np.searchsorted(model[0], T, side="left")
        out[strata == b] = np.where(exceed >= 10, (exceed + 1) / (model[1] + 1), surv(T, model, F))
    return out


def rule_sites(modified_peptide, mz, charge, shared_prefix, shared_suffix):
    """Number of junction-specific bond sites with a fragment within 0.02 Da (1+, and 2+ when z >= 3)."""
    hit_b, hit_y = matched_ions(modified_peptide, charge, mz[mz > 0])
    return len(specific_sites(parse_modified(modified_peptide)[0], hit_b, hit_y, shared_prefix, shared_suffix))


@functools.lru_cache(maxsize=6)
def shard_arrays(shard):
    meta = pd.read_parquet(f"{shard}/meta.parquet", columns=["cand_peptide"])
    return meta.cand_peptide.values, np.load(f"{shard}/mz.npy", mmap_mode="r"), np.load(f"{shard}/prec.npy")


def psm_source(world, shard, score, E):
    """Statistics of one scorer in one world: T of a peptide = its best top-1 score over the kept spectra."""
    s = np.where(world["cmask"], score, -np.inf)
    rows = np.arange(len(s))
    k = np.argmax(s, 1)
    best = s[rows, k]
    ok = np.isfinite(best)
    top1 = pd.DataFrame({"pep": world["cand"][rows, k][ok], "T": best[ok], "decoy": world["cdecoy"][rows, k][ok],
                         "i": np.flatnonzero(ok), "k": k[ok]})
    novel = top1[top1.pep.isin(set(E.entry) | set(E.decoy_seq))]
    novel_pentr = (~novel.decoy.values) & novel.pep.isin(set(E.entry[E.kind == "PENTR"])).values
    best_rows = top1[~top1.decoy].sort_values("T", ascending=False).drop_duplicates("pep")
    best_rows = best_rows[best_rows.pep.isin(E.entry)]
    return dict(decoys=top1[top1.decoy].groupby("pep")["T"].max(), targets=top1[~top1.decoy].groupby("pep")["T"].max(),
                ks=(novel["T"].values[novel.decoy.values], novel["T"].values[novel_pentr]),
                best={p: (shard, i, slot) for p, i, slot in zip(best_rows.pep, best_rows.i, best_rows.k)})


def entry_source(unit_dir, stats_path, E, shards):
    """Statistics of one scorer in one unit: T = the set statistic; the rule uses the set's first top-1 token."""
    U = load_unit(unit_dir)
    S = U["sets"].assign(stat=pd.read_parquet(stats_path).stat.values)
    top1 = np.flatnonzero(U["tokens"][:, TOKEN_FEATS.index("top1")] > 0.5)
    first_top1 = pd.Series(top1).groupby(np.searchsorted(U["ptr"], top1, side="right") - 1).first()
    refs = U["tok_ref"][first_top1.values].tolist()
    decoy_world = (S.kind == "DECOY").values & S.entry.isin(set(E.decoy_seq)).values
    return dict(decoys=S[S.kind == "DECOY"].set_index("entry").stat,
                targets=S[S.kind != "DECOY"].set_index("entry").stat,
                ks=(S.stat.values[decoy_world], S.stat.values[(S.kind == "PENTR").values]),
                best={S.entry.iat[j]: (shards[w], i, slot) for j, (w, i, slot) in zip(first_top1.index, refs)})


def scale_map(source, E, drop=None, extra=None):
    """x = -ln S(T | L) fitted on the decoys (without drop), of the entries or those of extra; no hit: 0."""
    decoys = source["decoys"]
    pool = decoys if drop is None else decoys[~decoys.index.isin(drop)]
    pool_strata = length_bin(pool.index.str.len().values)
    mapping = stratum_map(pool_strata, MIN_STRATUM)
    pool_strata = mapping[pool_strata]
    models, F, _ = models_with_validation(pool.values, pool_strata, 1000)
    x_pool = pd.Series(-np.log(apply_surv(pool.values, pool_strata, models, F)), index=pool.index)

    def x_of(T, strata):
        out = np.zeros(len(T))
        hit = np.isfinite(T)
        out[hit] = -np.log(apply_surv(T[hit], mapping[strata[hit]], models, F))
        return out

    if extra is not None:
        return dict(pool=x_pool, x_extra=x_of(decoys.reindex(extra.decoy).values, extra.lb.values))
    return dict(pool=x_pool, x_t=x_of(source["targets"].reindex(E.entry).values, E.lb.values),
                x_d=x_of(decoys.reindex(E.decoy_seq).values, E.lb.values))


def aggregate_null(per, null):
    """Null of A: the class decoys through the same maximum over worlds. Returns (pool, stratum map, models, F)."""
    U = pd.concat({i: p["pool"] for i, p in enumerate(per)}, axis=1).fillna(0.0)
    U = U[U.index.isin(null)]
    pool = U.values.max(1)
    strata = length_bin(pd.Series(U.index).str.len().values)
    mapping = stratum_map(strata, MIN_STRATUM)
    models, F, _ = models_with_validation(pool, mapping[strata], 1000)
    return pool, mapping, models, F


def p_values(A, A_decoy, lb, reference, mapping, models, F):
    """p = pi_hit(L) x S_A(A | L); pi_hit = the fraction of the reference entries' decoys with a hit, per length bin."""
    hit = hit_rate(A_decoy[reference] > 0, lb[reference])
    pi_hit = pd.Series(lb).map(hit).fillna(hit.mean()).values
    return np.where(A > 0, np.minimum(1, pi_hit * final_survival(A, mapping[lb], models, F)), 1.0)


def truncated_ks(decoy, entrapment):
    """KS p of the lower halves of the two score lists, where no true entry is expected."""
    median = np.median(np.r_[decoy, entrapment]) if len(decoy) + len(entrapment) else 0
    low_decoy, low_entr = decoy[decoy <= median], entrapment[entrapment <= median]
    return float(ks_2samp(low_decoy, low_entr).pvalue) if min(len(low_decoy), len(low_entr)) > 10 else np.nan


def evaluate_fold(name, fold, sources, E, novel):
    """One scorer on one fold. Returns result rows, the entry table and the uniformity rows."""
    per = [scale_map(source, E) for source in sources]
    null = set(novel.decoy)
    calib = []   # class-decoy uniformity: decoys of root half r get p from the pool without the novel decoys of half r
    for half in (0, 1):
        held_out = novel[novel.half == half].reset_index(drop=True)
        half_per = [scale_map(source, E, drop=set(held_out.decoy), extra=held_out) for source in sources]
        _, mapping, models, F = aggregate_null(half_per, null)
        A = np.c_[[p["x_extra"] for p in half_per]].T.max(1)
        p = final_survival(A[A > 0], mapping[held_out.lb.values[A > 0]], models, F)
        calib.append(dict(scorer=name, n_eval=int((A > 0).sum()), **{f"k_{t:g}": int((p <= t).sum()) for t in CUTOFFS}))
    XT = np.c_[[p["x_t"] for p in per]].T
    A_target = XT.max(1)
    A_decoy = np.c_[[p["x_d"] for p in per]].T.max(1)
    pool, mapping, models, F = aggregate_null(per, null)
    is_class = E.kind.isin(CLASS).values
    class_idx = np.flatnonzero(is_class)
    p = p_values(A_target, A_decoy, E.lb.values, is_class, mapping, models, F)
    ks_p = truncated_ks(np.concatenate([source["ks"][0] for source in sources]),
                        np.concatenate([source["ks"][1] for source in sources]))
    ends = pd.read_csv(f"{RESULTS}/optionb_v3/fold{fold}/shared_ends.tsv", sep="\t").set_index("entry")

    @functools.cache
    def rule_ok(j):
        """Neighbour rule on the entry's best PSM in the world or unit attaining A."""
        entry = E.entry.iat[j]
        ref = sources[int(np.argmax(XT[j]))]["best"].get(entry)
        if ref is None or entry not in ends.index:
            return False
        candmod, mz, prec = shard_arrays(ref[0])
        return rule_sites(candmod[ref[1]][ref[2]], np.asarray(mz[ref[1]]), int(prec[ref[1], 1]),
                          int(ends.at[entry, "L_pre"]), int(ends.at[entry, "L_suf"])) >= 2

    rows = []
    for alpha in (0.01, 0.05):
        rejected = bh(p[is_class], alpha)
        found_idx = class_idx[rejected]
        pentr_idx = np.flatnonzero((E.kind.values == "PENTR") & (p <= p[found_idx].max(initial=0.0)))
        found = E.iloc[found_idx]
        kept = E.iloc[found_idx[np.array([rule_ok(j) for j in found_idx], bool)]]
        rows.append(dict(scorer=name, fold=f"fold{fold}", alpha=alpha, discoveries=len(found),
                         true=int(found.true.sum()), known_false=int(found.kind.isin(KNOWN_FALSE).sum()),
                         unknown=int(((found.kind == "PSEUDO") & ~found.true).sum()), pentr=len(pentr_idx),
                         true_available=int(E.true.sum()), rule_discoveries=len(kept), rule_true=int(kept.true.sum()),
                         rule_known_false=int(kept.kind.isin(KNOWN_FALSE).sum()),
                         rule_unknown=int(((kept.kind == "PSEUDO") & ~kept.true).sum()),
                         rule_pentr=sum(rule_ok(j) for j in pentr_idx), ks_p=ks_p, F_agg=F, n_pool_union=len(pool)))
    top = class_idx[np.lexsort((-A_target[class_idx], p[class_idx]))][:2000]   # the rule depth of the TKF curves
    passed = np.full(len(E), np.nan)
    passed[top] = [float(rule_ok(j)) for j in top]
    return rows, E.assign(scorer=name, fold=f"fold{fold}", A=A_target, p=p, rule_ok=passed), calib


def tkf_curve(d, use_rule=True):
    """Known-truth curve: class entries of all folds ranked by p (ties: larger A first), optionally only rule passes."""
    d = d[d.kind.isin(CLASS)]
    if use_rule:
        d = d[d.rule_ok == 1.0]
    d = d.iloc[np.lexsort((-d.A.values, d.p.values))]
    depth = np.arange(1, len(d) + 1)
    return pd.DataFrame({"depth": depth, "T": np.cumsum(d.true.values.astype(int)),
                         "KF": np.cumsum(d.kind.isin(KNOWN_FALSE).values.astype(int)), "D": depth, "p": d.p.values})


def tkf_value(d, q, use_rule=True):
    """TKF_q: true discoveries at known-false FDP <= q."""
    curve = tkf_curve(d, use_rule)
    ok = curve.KF.values <= q * curve.D.values
    return int(curve["T"].values[ok].max()) if ok.any() else 0


def boot_gain(ET, a, b):
    """Paired bootstrap over junction groups of true(a) - true(b) at class FDR 1 % before the rule."""
    rng = np.random.default_rng(BOOT_SEED)
    groups = ET[ET.scorer == a][["fold", "jkey"]].drop_duplicates().reset_index(drop=True)
    tables = [(sign, ET[ET.scorer == x].set_index(["fold", "jkey"])) for sign, x in ((1, a), (-1, b))]
    gains = []
    for _ in range(N_BOOT):
        sample = groups.iloc[rng.integers(0, len(groups), len(groups))]
        gain = 0
        for sign, table in tables:
            resampled = table.loc[list(zip(sample.fold, sample.jkey))]
            for _, fold_table in resampled.groupby(level=0):
                is_class = fold_table.kind.isin(CLASS).values
                gain += sign * int(fold_table[is_class][bh(fold_table.p.values[is_class], 0.01)].true.sum())
        gains.append(gain)
    return np.percentile(gains, [2.5, 50, 97.5]).tolist()


def boot_tkf(ET, a, b, q):
    """Paired bootstrap over junction groups of TKF_q(a) - TKF_q(b), resampling rows by position."""
    rng = np.random.default_rng(BOOT_SEED)
    table_a, table_b = ET[ET.scorer == a], ET[ET.scorer == b]
    groups = table_a[["fold", "jkey"]].drop_duplicates().reset_index(drop=True)
    group_of = {key: i for i, key in enumerate(zip(groups.fold, groups.jkey))}
    index_a = np.array([group_of[key] for key in zip(table_a.fold, table_a.jkey)])
    index_b = np.array([group_of.get(key, -1) for key in zip(table_b.fold, table_b.jkey)])
    gains = []
    for _ in range(N_BOOT):
        weights = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
        sample_a = table_a.iloc[np.repeat(np.arange(len(table_a)), weights[index_a])]
        sample_b = table_b.iloc[np.repeat(np.arange(len(table_b)), np.where(index_b >= 0, weights[index_b], 0))]
        gains.append(tkf_value(sample_a, q) - tkf_value(sample_b, q))
    return [float(np.quantile(gains, 0.025)), float(np.median(gains)), float(np.quantile(gains, 0.975))]


def write_outputs(rows, entry_tables, calib, out, pairs, baseline):
    os.makedirs(out, exist_ok=True)
    ET = pd.concat(entry_tables)
    pd.DataFrame(rows).to_csv(f"{out}/results.tsv", sep="\t", index=False)
    ET.to_csv(f"{out}/entries.tsv.gz", sep="\t", index=False)
    pooled = []   # k against the two-sided 95 % binomial acceptance interval at each cutoff
    for name, d in pd.DataFrame(calib).groupby("scorer"):
        n = int(d.n_eval.sum())
        row = dict(scorer=name, pool="a8", n_eval=n)
        for t in CUTOFFS:
            k = int(d[f"k_{t:g}"].sum())
            lo, hi = int(binom.ppf(0.025, n, t)), int(binom.ppf(0.975, n, t))
            row.update({f"k_{t:g}": k, f"exp_{t:g}": round(n * t, 2), f"lo_{t:g}": lo, f"hi_{t:g}": hi,
                        f"ok_{t:g}": lo <= k <= hi})
        pooled.append(dict(row, uniform_pass=all(row[f"ok_{t:g}"] for t in CUTOFFS)))
    pd.DataFrame(pooled).to_csv(f"{out}/calib_pooled.tsv", sep="\t", index=False)
    tkf, curves, gains = [], [], []
    for name, d in ET.groupby("scorer"):
        tkf.append(dict(scorer=name, **{f"TKF_{q:g}_{tag}": tkf_value(d, q, use_rule) for q in (0.02, 0.05)
                                        for tag, use_rule in (("rule", True), ("norule", False))}))
        curves += [tkf_curve(d, use_rule).assign(scorer=name, rule=use_rule) for use_rule in (True, False)]
        for fold, table in d.groupby("fold"):
            is_class = table.kind.isin(CLASS).values
            for threshold in (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1):
                rejected = bh(table.p.values[is_class], threshold)
                cut = table.p.values[is_class][rejected].max(initial=0.0)
                found = table[is_class][rejected]
                gains.append(dict(scorer=name, fold=fold, threshold=threshold, n_true=int(found.true.sum()),
                                  n_disc=int(rejected.sum()), n_known_false=int(found.kind.isin(KNOWN_FALSE).sum()),
                                  pentr=int((table.p.values[table.kind.values == "PENTR"] <= cut).sum())))
    pd.DataFrame(tkf).to_csv(f"{out}/tkf.tsv", sep="\t", index=False)
    curves = pd.concat(curves)
    curves[curves.depth <= 3000].to_csv(f"{out}/tkf_curve.tsv.gz", sep="\t", index=False)
    pd.DataFrame(gains).to_csv(f"{out}/gain.source.tsv", sep="\t", index=False)
    if pairs:
        with_baseline = [] if out == baseline else [pd.read_csv(f"{baseline}/entries.tsv.gz", sep="\t")]
        joined = pd.concat([ET] + with_baseline, ignore_index=True)
        results = dict(boot={f"{a}:{b}": boot_gain(joined, a, b) for a, b in pairs},
                       boot_tkf={f"{a}:{b}@{q:g}": boot_tkf(joined, a, b, q) for a, b in pairs for q in (0.02, 0.05)})
        with open(f"{out}/results.json", "w") as fh:
            json.dump(results, fh, indent=1, default=float)


def psm_scores(world, cohort, fold, trunc):
    """Sage LDA, B2 (undefined on truncated sets) and v1x, the stitched stage-1 scores."""
    scores = {} if trunc else {"B2": np.load(f"{RESULTS}/b2/v3_{cohort}_f{fold}/scores.npy").astype(np.float64)}
    scores["sage_lda"] = world["cfeat"][:, :, 10].astype(np.float64)
    scores["v1x"] = np.load(f"{RESULTS}/runs/stitched/v3_{cohort}_f{fold}{trunc}/scores.npy").astype(np.float64)
    return scores


def evaluate_set(model, set_name):
    """model = psm (every PSM scorer) or an ET round, with the PSM scorers of the same set as paired baselines."""
    folds, psm_out, et_out, prefix, kind = SETS[set_name]
    trunc = "_trunc" if kind == "full_trunc" else ""
    rows, tables, calib = [], [], []
    for fold in folds:
        entries = fold_entries(fold)
        entries = entries.assign(decoy_seq=entries.entry.map(sage_decoy), lb=length_bin(entries.entry.str.len().values))
        root = np.where(entries.kind.isin(["PSEUDO", "PENTR"]), entries.jkey, entries.entry)
        novel = pd.DataFrame({"decoy": entries.decoy_seq, "lb": entries.lb, "half": [crc(r) % 2 for r in root]})
        E = entries[entries.in_class | (entries.kind == "PENTR")].reset_index(drop=True)
        shards = [f"{RESULTS}/shards_v3/{c}_f{fold}_eval" + ("_trunc_full" if trunc else "") for c in OPTIONB_COHORTS]
        worlds = [load_world([shard]) for shard in shards]
        E["true"] = (E.kind == "PSEUDO") & E.entry.map(il).isin(set().union(*[world["truth"] for world in worlds]))
        if model == "psm":
            scores = [psm_scores(world, cohort, fold, trunc) for world, cohort in zip(worlds, OPTIONB_COHORTS)]
            sources = {name: [psm_source(world, shard, score[name], E)
                              for world, shard, score in zip(worlds, shards, scores)] for name in scores[0]}
        else:
            stats = f"{RESULTS}/entry/rounds/{model}/{prefix}_f{fold}_h"
            sources = {ET_NAMES[model]: [entry_source(f"{RESULTS}/entry/units/{kind}/f{fold}_h{h}",
                                                      f"{stats}{h}/stats.parquet", E,
                                                      [x[0][0] for x in fold_worlds(fold, h, trunc)]) for h in (0, 1)]}
        for name, source_list in sources.items():
            fold_rows, table, fold_calib = evaluate_fold(name, fold, source_list, E, novel)
            rows += fold_rows
            tables.append(table)
            calib += fold_calib
    baseline = f"{RESULTS}/eval_v2/{psm_out}"
    pairs = [("B2", "v1x"), ("B2", "sage_lda"), ("v1x", "sage_lda")]
    if model != "psm":
        pairs = [(ET_NAMES[model], x) for x in ("B2", "sage_lda", "v1x")]
    out = baseline if model == "psm" else f"{RESULTS}/entry/rounds/{model}/{et_out}"
    write_outputs(rows, tables, calib, out, [] if trunc else pairs, baseline)


def evaluate_production(round_name):
    """Production units: x per unit from the mean logit of the models, one BH over the disc entries, the sentinels."""
    out = f"{RESULTS}/prod_eval/{round_name}"
    os.makedirs(out, exist_ok=True)
    E = prod_entries()
    E["decoy_seq"] = E.entry.map(sage_decoy)
    E["lb"] = length_bin(E.entry.str.len().values)
    X = []
    for half in (0, 1):
        S = pd.read_parquet(f"{RESULTS}/entry/units/prod/prod_h{half}/sets.parquet")
        stats = [f"{RESULTS}/entry/rounds/{round_name}/prod_score_g{g}h{half}/stats.parquet" for g in (0, 1)]
        S["stat"] = np.mean([pd.read_parquet(path).stat.values for path in stats], 0)
        decoys = S[S.kind == "DECOY"]
        strata = length_bin(decoys.entry.str.len().values)
        mapping = stratum_map(strata, MIN_STRATUM)
        models, F, _ = models_with_validation(decoys.stat.values, mapping[strata], 1000)
        all_strata = mapping[length_bin(S.entry.str.len().values)]
        X.append(pd.Series(-np.log(apply_surv(S.stat.values, all_strata, models, F)), index=S.entry.values))
    A_target = np.c_[[x.reindex(E.entry).fillna(0).values for x in X]].T.max(1)
    A_decoy = np.c_[[x.reindex(E.decoy_seq).fillna(0).values for x in X]].T.max(1)
    _, mapping, models, F_agg = aggregate_null([dict(pool=x) for x in X], set(E.decoy_seq))
    is_chim = (E.kind == "CHIM").values
    E["A"] = A_target
    E["p"] = p_values(A_target, A_decoy, E.lb.values, is_chim, mapping, models, F_agg)
    E["A_decoy"] = A_decoy
    E["p_decoy"] = p_values(A_decoy, A_decoy, E.lb.values, is_chim, mapping, models, F_agg)
    main_hits = pd.read_csv("results/cptac/pan/main_hits.tsv", sep="\t")
    E["main_hit"] = E.entry.isin(set(main_hits.peptide[main_hits.status == "main"])) & (E.kind == "CHIM")
    disc = is_chim & E.in_class.values
    entr = (E.kind == "ENTR").values & E.family.eq("disc").values
    # paired scores: each ENTR entry against its target (Wen et al. 2025, r = 1)
    target_score = -np.log10(E.set_index("entry").p.reindex(E.target_of[entr]).fillna(1.0).values.clip(1e-300))
    entr_score = -np.log10(E.p.values[entr].clip(1e-300))
    rows, lists = [], []
    for alpha in (0.01, 0.05):
        rejected = bh(E.p.values[disc], alpha)
        cut = E.p.values[disc][rejected].max(initial=0.0)
        found = E[disc][rejected].assign(alpha=alpha)
        lists.append(found)
        n_target = len(found)
        n_entr = int((entr & (E.p.values <= cut)).sum()) if len(found) else 0
        added = int((~found.main_hit).sum())
        log_cut = -np.log10(cut) if cut > 0 else np.inf
        target_found, entr_found = target_score >= log_cut, entr_score >= log_cut
        paired = int(entr_found.sum()) + int((entr_found & ~target_found).sum())
        paired += 2 * int((entr_found & target_found & (entr_score > target_score)).sum())
        rows.append(dict(alpha=alpha, discoveries=n_target, entr_under_cut=n_entr,
                         wen_eq1_combined=2 * n_entr / max(n_target + n_entr, 1),
                         wen_eq4_paired=paired / max(int(target_found.sum() + entr_found.sum()), 1),
                         chim_decoys_under_cut=int((E.p_decoy.values[is_chim] <= cut).sum()) if len(found) else 0,
                         increment_fdp_est=n_entr * 2 / max(added + n_entr, 1)))
    calibration = {f"chim_decoy_frac_p_le_{t:g}": float((E.p_decoy.values[is_chim] <= t).mean()) for t in CUTOFFS}
    calibration.update({f"entr_frac_p_le_{t:g}": float((E.p.values[entr] <= t).mean()) for t in CUTOFFS})
    decoy_A = A_decoy[is_chim][A_decoy[is_chim] > 0]
    entr_A = A_target[entr][A_target[entr] > 0]
    calibration["ks_p_decoy_vs_entr"] = truncated_ks(decoy_A, entr_A)
    E.to_csv(f"{out}/entries.tsv.gz", sep="\t", index=False)
    pd.concat(lists, ignore_index=True).to_csv(f"{out}/discoveries.tsv", sep="\t", index=False)
    pd.DataFrame(rows).to_csv(f"{out}/sentinels.tsv", sep="\t", index=False)
    pd.DataFrame([calibration]).to_csv(f"{out}/calib.tsv", sep="\t", index=False)


def sibling_table():
    """On confident dev spectra with an NNEG target among the candidates: top-1 is the true candidate or the sibling."""
    rows = []
    for fold in SETS["dev_f0123"][0]:
        for cohort in OPTIONB_COHORTS:
            shard = f"{RESULTS}/shards_v3/{cohort}_f{fold}_eval"
            world = load_world([shard])
            cm, dec = world["cmask"], world["cdecoy"]
            cls = np.load(f"{shard}/ccls.npy")   # 2 = NNEG
            label = np.load(f"{shard}/label.npy").astype(int)
            idx = np.flatnonzero((label >= 0) & (label < cm.shape[1]) & ((cls == 2) & ~dec & cm).any(1))
            scores = psm_scores(world, cohort, fold, "")
            for name in ("sage_lda", "v1x", "B2"):
                k = np.argmax(np.where(cm[idx], scores[name][idx], -np.inf), 1)
                rows.append(dict(fold=f"fold{fold}", cohort=cohort, scorer=name, n=len(idx),
                                 n_top1_true=int((k == label[idx]).sum()),
                                 n_top1_nneg=int(((cls[idx, k] == 2) & ~dec[idx, k]).sum())))
    R = pd.DataFrame(rows)
    counts = ["n", "n_top1_true", "n_top1_nneg"]
    per_fold = R.groupby(["fold", "scorer"])[counts].sum().reset_index().assign(cohort="ALL")
    pooled = R.groupby("scorer")[counts].sum().reset_index().assign(cohort="ALL", fold="ALL")
    R = pd.concat([R, per_fold, pooled], ignore_index=True)
    R["frac_top1_true"] = R.n_top1_true / R.n
    R["frac_top1_nneg"] = R.n_top1_nneg / R.n
    R.to_csv(f"{RESULTS}/e2/e2_v3_dev_f0123.tsv", sep="\t", index=False)


if __name__ == "__main__":
    if sys.argv[1] == "e2":
        sibling_table()
    elif sys.argv[1] == "prod":
        evaluate_production(sys.argv[2])
    else:
        evaluate_set(*sys.argv[1:])
