"""Entry level: token and set builder (a unit is one (fold, plex half) or production plex half) and set transformer on
the candidate rows of one entry, CPU, fp32. build_rows() never reads class or decoy labels. usage: python -m juncta_t.
model_entry unit full|full_trunc <fold> <half> | unit prod <half> | train <round> <g> <h> <seed> | score <round>"""
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from juncta.config import COHORTS, length_bin
from juncta.stats import apply_surv, models_with_validation
from juncta_t.common import CONFIG, FEATS, N_STRATA, OPTIONB_COHORTS, RESULTS, crc, il, sage_decoy, stratum_map

HP = CONFIG["entry_model"]
ROUNDS = f"{RESULTS}/entry/rounds"
UNITS = f"{RESULTS}/entry/units"
# The order is part of the trained models.
TOKEN_FEATS = (["s1", "r0", "r1", "r2", "r3", "r4", "top1", "margin", "none_lo", "x_psm"] + ["sage_" + s for s in FEATS]
               + ["z2", "z3", "z4", "z5p"])
SET_FEATS = [f"lb{b}" for b in range(N_STRATA)] + ["log_ntok", "log_ntop1", "n_coh_top1", "n_chg_top1"]
MIN_DECOYS_PER_FIT = 102   # the split-half validation fit needs at least 51 decoys per half


def fold_entries(fold):
    """Hypotheses of a fold; the class is PSEUDO + REJ + the n_cap NNEG and n_cap_mod MOD entries of smallest crc32."""
    with open(f"{RESULTS}/siblings/caps_per_fold.json") as fh:
        caps = json.load(fh)[str(fold)]
    pseudo, pentr, nneg, rej, mod = (pd.read_csv(f"{RESULTS}/optionb_v3/fold{fold}/{name}.tsv", sep="\t")
                                     for name in ("pseudo", "pentr", "nneg", "rej", "mod"))
    junction = dict(zip(pseudo.peptide, pseudo.junction_key))

    def capped(table, kind, cap, prefix):
        entries = pd.Series(sorted(table.peptide, key=crc), dtype=object)
        return pd.DataFrame({"entry": entries, "kind": kind, "in_class": entries.index < cap, "jkey": prefix + entries})

    parts = [pd.DataFrame({"entry": pseudo.peptide, "kind": "PSEUDO", "in_class": True, "jkey": pseudo.junction_key}),
             pd.DataFrame({"entry": rej.peptide, "kind": "REJ", "in_class": True, "jkey": "R:" + rej.peptide}),
             capped(nneg, "NNEG", caps["n_cap"], "N:"), capped(mod, "MOD", caps["n_cap_mod"], "M:"),
             pd.DataFrame({"entry": pentr.peptide, "kind": "PENTR", "in_class": False,
                           "jkey": pentr.target.map(junction).fillna("E:" + pentr.peptide)})]
    return pd.concat(parts, ignore_index=True).drop_duplicates("entry").reset_index(drop=True)


def prod_entries():
    """Production hypotheses: CHIMPEP entries of every family (in_class = disc, the BH family of 47,975) and ENTR."""
    path = "results/cptac/pan/analysis/PDC000110/pvals_gpd.tsv.gz"   # the DB is identical in every cohort
    table = pd.read_csv(path, sep="\t", usecols=["cls", "decoy", "entry", "family", "target_of"], low_memory=False)
    table = table[~table.decoy & table.cls.isin(["CHIM", "ENTR"])].drop_duplicates("entry").reset_index(drop=True)
    return table.assign(kind=table.cls, in_class=(table.cls == "CHIM") & (table.family == "disc"), jkey=table.entry)


def load_world(shards):
    """One world = one shard or the chunk shards of one cohort-half. truth = confident reference labels (I = L)."""
    columns = ["orig_q", "orig_stripped", "orig_cls", "cand_stripped"]
    meta = pd.concat([pd.read_parquet(f"{shard}/meta.parquet", columns=columns) for shard in shards], ignore_index=True)
    world = {name: np.concatenate([np.load(f"{shard}/{name}.npy") for shard in shards])
             for name in ("cmask", "cdecoy", "prec", "cfeat")}
    cand = np.full(world["cmask"].shape, "", dtype=object)
    for i, row in enumerate(meta.cand_stripped.values):
        cand[i, :len(row)] = row
    confident = (meta.orig_q.fillna(1).values <= 0.01) & (meta.orig_cls.fillna("").values == "ref")
    return dict(world, cand=cand, truth={il(x) for x in meta.orig_stripped.values[confident]})


def build_rows(cm, cand, s1, logits, cfeat, prec, calibration, want):
    """Token features of the candidate rows with cand in want, with their row, slot, peptide, top1 flag, log-odds."""
    n, K = cm.shape
    s = np.where(cm, s1, -np.inf)
    order = np.argsort(-s, 1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(K)[None, :].repeat(n, 0), 1)
    ranked = np.take_along_axis(s, order, 1)
    best, second = ranked[:, 0], ranked[:, 1]
    ii, kk = np.nonzero(cm)
    wanted = np.fromiter((p in want for p in cand[ii, kk]), bool, len(ii))
    ii, kk = ii[wanted], kk[wanted]
    pep = cand[ii, kk]
    value = s[ii, kk]
    row_rank = rank[ii, kk]
    top1 = row_rank == 0
    other = np.where(top1, second[ii], best[ii])
    margin = np.where(np.isfinite(other), value - other, value)
    lg = np.where(np.c_[cm, np.ones((n, 1), bool)], logits, -1e4)
    none_log_odds = (lg[:, K] - np.logaddexp.reduce(lg[:, :K], axis=1))[ii]
    models, mapping, F = calibration
    lengths = np.fromiter((len(p) for p in pep), int, len(pep))
    x_psm = -np.log(np.clip(apply_surv(value, mapping[length_bin(lengths)], models, F), 1e-300, 1))
    charge = prec[ii, 1].astype(int)
    feat = np.zeros((len(ii), len(TOKEN_FEATS)), np.float32)
    feat[:, 0] = np.clip(value, -30, 30)
    feat[np.arange(len(ii)), 1 + np.clip(row_rank, 0, 4)] = 1
    feat[:, 6] = top1
    feat[:, 7] = np.clip(margin, -30, 30)
    feat[:, 8] = np.clip(none_log_odds, -30, 30)
    feat[:, 9] = np.clip(x_psm, 0, 50)
    feat[:, 10:30] = np.nan_to_num(np.asarray(cfeat[ii, kk], np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    feat[:, 30:34] = np.c_[charge <= 2, charge == 3, charge == 4, charge >= 5]
    return feat, ii, kk, pep, top1, value


def build_unit(E, worlds, out):
    """Sets of the entries and of the Sage decoys of the worlds; only sets with a stage-1 top-1 token are written."""
    feats, refs, parts = [], [], []
    truth = set()
    for wi, (shards, score_dirs) in enumerate(worlds):
        W = load_world(shards)
        s1, logits = (np.concatenate([np.load(f"{d}/{name}.npy") for d in score_dirs]).astype(np.float64)
                      for name in ("scores", "logits"))
        truth |= W["truth"]
        # per-world survival per length stratum over the best top-1 log-odds of every decoy peptide winning a spectrum
        s = np.where(W["cmask"], s1, -np.inf)
        rows, k = np.arange(len(s)), np.argmax(s, 1)
        decoy_top1 = W["cdecoy"][rows, k] & np.isfinite(s[rows, k])
        pool = pd.Series(s[rows, k][decoy_top1]).groupby(W["cand"][rows, k][decoy_top1]).max()
        strata = length_bin(pool.index.str.len().values)
        mapping = stratum_map(strata, MIN_DECOYS_PER_FIT)
        models, F, _ = models_with_validation(pool.values, mapping[strata], 1000)
        want = set(W["cand"][W["cmask"] & W["cdecoy"]].tolist()) | set(E.entry)
        f, ii, kk, pep, top1, value = build_rows(W["cmask"], W["cand"], s1, logits, W["cfeat"], W["prec"],
                                                 (models, mapping, F), want)
        feats.append(f)
        refs.append(np.c_[np.full(len(ii), wi), ii, kk].astype(np.int32))
        charge = W["prec"][ii, 1].astype(np.int8)
        parts.append(pd.DataFrame({"pep": pep, "s1": value, "top1": top1, "coh": wi, "chg": charge}))
    R = pd.concat(parts, ignore_index=True).reset_index(names="r")
    hit = R[R.top1].groupby("pep").agg(n_top1=("r", "size"), n_coh=("coh", "nunique"), n_chg=("chg", "nunique"))
    R = R[R.pep.isin(hit.index)].sort_values(["pep", "s1"], ascending=[True, False], kind="stable")
    n_tok = R.groupby("pep", sort=False).size()
    sets = pd.DataFrame({"entry": n_tok.index, "n_tok": n_tok.values}).join(hit, on="entry")
    sets["kind"] = sets.entry.map(E.set_index("entry").kind).fillna("DECOY")
    # the root of a PSEUDO or PENTR entry is its junction; a decoy follows its target
    root_of = dict(zip(E.entry, np.where(E.kind.isin(["PSEUDO", "PENTR"]), E.jkey, E.entry)))
    target = sets.entry.map({sage_decoy(e): e for e in E.entry}).where(sets.kind == "DECOY").fillna(sets.entry)
    sets["root"] = target.map(root_of).fillna(sets.entry)
    sets["true"] = (sets.kind == "PSEUDO") & sets.entry.map(il).isin(truth)
    gfeat = np.zeros((len(sets), len(SET_FEATS)), np.float32)
    gfeat[np.arange(len(sets)), length_bin(sets.entry.str.len().values)] = 1
    gfeat[:, N_STRATA] = np.log1p(sets.n_tok.values)
    gfeat[:, N_STRATA + 1] = np.log1p(sets.n_top1.values)
    gfeat[:, N_STRATA + 2] = sets.n_coh.values
    gfeat[:, N_STRATA + 3] = sets.n_chg.values
    os.makedirs(out, exist_ok=True)
    np.save(f"{out}/tokens.npy", np.concatenate(feats)[R.r.values])
    np.save(f"{out}/ptr.npy", np.r_[0, np.cumsum(sets.n_tok.values)].astype(np.int64))
    np.save(f"{out}/gfeat.npy", gfeat)
    np.save(f"{out}/tok_ref.npy", np.concatenate(refs)[R.r.values])
    sets[["entry", "kind", "root", "true"]].to_parquet(f"{out}/sets.parquet", index=False)


def load_unit(directory):
    U = {name: np.load(f"{directory}/{name}.npy") for name in ("tokens", "ptr", "gfeat", "tok_ref")}
    return dict(U, sets=pd.read_parquet(f"{directory}/sets.parquet"))


def fold_worlds(fold, half, suffix=""):
    """The eval half-shards scored by the stage-1 model that saw neither the fold's genes nor the half."""
    name = f"f{fold}_eval_h{half}{suffix}"
    run = f"{RESULTS}/runs/s1_g{1 - fold % 2}h{1 - half}"
    return [([f"{RESULTS}/shards_v3/{c}_{name}"], [f"{run}/score_v3_{c}_{name}_fp32"]) for c in OPTIONB_COHORTS]


def prod_worlds(half):
    """All chunk shards of each cohort in plex half h, scored by S_{0,1-h}."""
    worlds = []
    for cohort in sorted(COHORTS):
        chunks = glob.glob(f"{RESULTS}/prod/shards/{cohort}_h{half}_c*")
        chunks.sort(key=lambda x: int(x.rsplit("_c", 1)[1]))
        worlds.append((chunks, [f"{RESULTS}/runs/prod_s1/score_prod_{os.path.basename(x)}_fp32" for x in chunks]))
    return worlds


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        d = HP["d"]
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, HP["ff"]), nn.GELU(), nn.Dropout(HP["dropout"]), nn.Linear(HP["ff"], d))
        self.drop = nn.Dropout(HP["dropout"])

    def forward(self, x, attn_bias):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, HP["heads"], d // HP["heads"]).permute(2, 0, 3, 1, 4)
        dropout = HP["dropout"] if self.training else 0.0
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, T, d)))
        return x + self.drop(self.ff(self.ln2(x)))


class EntryTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d, n_tok, n_set = HP["d"], len(TOKEN_FEATS), len(SET_FEATS)
        self.register_buffer("tok_mu", torch.zeros(n_tok))
        self.register_buffer("tok_sd", torch.ones(n_tok))
        self.register_buffer("set_mu", torch.zeros(n_set))
        self.register_buffer("set_sd", torch.ones(n_set))
        self.tok_in = nn.Sequential(nn.Linear(n_tok, d), nn.GELU(), nn.Linear(d, d))
        self.set_in = nn.Sequential(nn.Linear(n_set, d), nn.GELU(), nn.Linear(d, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(HP["layers"])])
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)

    def forward(self, tok, mask, gfeat):
        B, T, _ = tok.shape
        x = self.tok_in(((tok - self.tok_mu) / self.tok_sd).clamp(-10, 10))
        c = self.cls.expand(B, 1, -1) + self.set_in(((gfeat - self.set_mu) / self.set_sd).clamp(-10, 10)).unsqueeze(1)
        h = torch.cat([c, x], 1)
        key_ok = torch.cat([torch.ones(B, 1, dtype=torch.bool), mask], 1)
        bias = torch.zeros(B, HP["heads"], T + 1, T + 1).masked_fill(~key_ok[:, None, None, :], float("-inf"))
        for block in self.blocks:
            h = block(h, bias)
        return self.out(self.ln(h[:, 0])).squeeze(-1)


def collate(tokens, ptr, gfeat, idx):
    """CSR gather of sets idx into padded tensors; tokens are in descending stage-1 order, so the cap keeps the best."""
    start = ptr[idx]
    length = np.minimum(ptr[idx + 1] - start, HP["cap"])
    positions = np.arange(max(int(length.max()), 1))[None, :]
    valid = positions < length[:, None]
    tok = np.zeros((len(idx), positions.size, tokens.shape[1]), np.float32)
    tok[valid] = tokens[(start[:, None] + positions)[valid]]
    set_features = np.ascontiguousarray(gfeat[idx], np.float32)
    return torch.from_numpy(tok), torch.from_numpy(valid), torch.from_numpy(set_features)


@torch.no_grad()
def predict(model, tokens, ptr, gfeat, idx, batch=2048):
    model.eval()
    out = np.zeros(len(idx), np.float64)
    order = np.argsort(ptr[idx + 1] - ptr[idx], kind="stable")   # length-sorted batches waste less padding
    for start in range(0, len(order), batch):
        j = order[start:start + batch]
        out[j] = model(*collate(tokens, ptr, gfeat, idx[j])).double().numpy()
    return out


def train(round_name, g, h, seed_index):
    """ET_{g,h} on half 1 - h of the folds of the other coarse group. Positives are true PSEUDO sets; negatives every
    Sage decoy and NNEG set, ETk also REJ and MOD; the epoch with the lowest weighted validation BCE is kept."""
    torch.set_num_threads(3)
    out = f"{ROUNDS}/{round_name}/ET_g{g}h{h}"
    os.makedirs(out, exist_ok=True)
    units = [load_unit(f"{UNITS}/full/f{fold}_h{1 - h}") for fold in range(1 - g, 6, 2)]
    tokens = np.concatenate([unit["tokens"] for unit in units])
    ptr = np.r_[0, np.cumsum(np.concatenate([np.diff(unit["ptr"]) for unit in units]))].astype(np.int64)
    gfeat = np.concatenate([unit["gfeat"] for unit in units])
    S = pd.concat([unit["sets"] for unit in units], ignore_index=True)
    group = np.full(len(S), "", dtype=object)
    group[(S.kind == "PSEUDO") & S.true] = "pos"
    negatives = ["DECOY", "NNEG"] + (["REJ", "MOD"] if round_name == "R1b_ETk" else [])
    group[S.kind.isin(negatives)] = S.kind[S.kind.isin(negatives)]
    y = (group == "pos").astype(np.float32)
    in_validation = np.array([crc(r + "|val") % 10 == 0 for r in S.root.values])
    train_idx = np.flatnonzero((group != "") & ~in_validation)
    val_idx = np.flatnonzero((group != "") & in_validation)

    def set_weights(idx):
        groups = pd.Series(group[idx])
        return (groups.map(HP["group_weights"]) / groups.map(groups.value_counts())).values

    set_weight = np.zeros(len(S))
    set_weight[train_idx] = set_weights(train_idx)
    val_weight = set_weights(val_idx)
    lengths = np.minimum(np.diff(ptr), HP["cap"])
    seed = HP["seed"] + seed_index
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = EntryTransformer()
    sampled_sets = train_idx[:: max(1, len(train_idx) // 200000)]
    sample = tokens[np.concatenate([np.arange(ptr[i], ptr[i + 1]) for i in sampled_sets])]
    sample = sample[np.random.default_rng(0).choice(len(sample), min(len(sample), 2_000_000), replace=False)]
    model.tok_mu.copy_(torch.from_numpy(sample.mean(0)))
    model.tok_sd.copy_(torch.from_numpy(sample.std(0) + 1e-3))
    model.set_mu.copy_(torch.from_numpy(gfeat[train_idx].mean(0)))
    model.set_sd.copy_(torch.from_numpy(gfeat[train_idx].std(0) + 1e-3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    best_loss = np.inf
    B = HP["batch"]
    for epoch in range(HP["epochs"]):
        model.train()
        perm = rng.permutation(train_idx)
        batches = []
        for c0 in range(0, len(perm), 64 * B):
            chunk = perm[c0:c0 + 64 * B]
            chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
            batches += [chunk[j:j + B] for j in range(0, len(chunk), B)]
        for j in rng.permutation(len(batches)):
            idx = batches[j]
            logit = model(*collate(tokens, ptr, gfeat, idx))
            w = torch.from_numpy(set_weight[idx].astype(np.float32))
            loss = F.binary_cross_entropy_with_logits(logit, torch.from_numpy(y[idx]), reduction="none")
            loss = (w * loss).sum() / w.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        logit = predict(model, tokens, ptr, gfeat, val_idx)
        val_loss = (val_weight * (np.logaddexp(0, logit) - y[val_idx] * logit)).sum() / val_weight.sum()
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(dict(state=model.state_dict(), epoch=epoch, seed=seed), f"{out}/model_s{seed_index}.pt")


def score(model_dir, unit_dir, out):
    """Mean logit over the seeds of one model."""
    U = load_unit(unit_dir)
    logits = []
    for path in sorted(glob.glob(f"{model_dir}/model_s*.pt")):
        model = EntryTransformer()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state"])
        logits.append(predict(model, U["tokens"], U["ptr"], U["gfeat"], np.arange(len(U["sets"]))))
    os.makedirs(out, exist_ok=True)
    U["sets"][["entry"]].assign(stat=np.mean(logits, 0)).to_parquet(f"{out}/stats.parquet", index=False)


def score_round(round_name):
    """Unit (fold, half) is scored by ET_{fold % 2, half}; a production half by both models of that half."""
    torch.set_num_threads(8)
    root = f"{ROUNDS}/{round_name}"
    for kind, prefix, folds in (("full", "score", range(4)), ("full", "test_score", (4, 5)),
                                ("full_trunc", "trunc_score", range(6))):
        for fold, h in itertools.product(folds, (0, 1)):
            score(f"{root}/ET_g{fold % 2}h{h}", f"{UNITS}/{kind}/f{fold}_h{h}", f"{root}/{prefix}_f{fold}_h{h}")
    for g, h in itertools.product((0, 1), (0, 1)):
        score(f"{root}/ET_g{g}h{h}", f"{UNITS}/prod/prod_h{h}", f"{root}/prod_score_g{g}h{h}")


if __name__ == "__main__":
    mode, name, *numbers = sys.argv[1:]
    numbers = [int(x) for x in numbers]
    if mode == "train":
        train(name, *numbers)
    elif mode == "score":
        score_round(name)
    elif name == "prod":
        build_unit(prod_entries(), prod_worlds(numbers[0]), f"{UNITS}/prod/prod_h{numbers[0]}")
    else:
        unit_worlds = fold_worlds(*numbers, "_trunc" if name == "full_trunc" else "")
        build_unit(fold_entries(numbers[0]), unit_worlds, f"{UNITS}/{name}/f{numbers[0]}_h{numbers[1]}")
