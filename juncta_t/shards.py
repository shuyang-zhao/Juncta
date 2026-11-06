"""Stage-1 data: spectra selected from the Sage searches, then numpy shards, one row per spectrum: mz, it [n, N]; prec
(neutral mass, charge, aligned RT); tok [n, K, 40]; cmask, cdecoy, ccls [n, K]; cfeat [n, K, F]; label (-1, slot or K).
usage: python -m juncta_t.shards fold <PDC> <fold> | pretrain <PDC> | production <PDC> | stitch <PDC> <fold>"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from juncta.spectra import mgf_blocks, parse_block, reduce_peaks
from juncta_t.common import CONFIG, FEATS, OB, OPTIONB_COHORTS, RESULTS, TOKENS, crc, il, plex_half
from juncta_t.worlds import accession_gene_map, gene_chromosomes, gene_of_accession

KEY = ["filename", "scan"]
OPTIONB_ROOT = f"{RESULTS}/optionb_v3"
SELECTION_ROOT = f"{RESULTS}/sel_v3"
SHARDS = f"{RESULTS}/shards_v3"
CLASS_CODE = {"ref": 0, "PSEUDO": 1, "NNEG": 2, "REJ": 3, "PENTR": 4, "CHIM": 5, "ENTR": 6, "MOD": 7}
NOVEL_CODES = (1, 2, 3, 4, 7)
N_PEAKS, K_CANDIDATES, L_MAX, PROTON = 150, 5, 40, 1.007276
PEAK_SCHEMA = pa.schema([("filename", pa.string()), ("scan", pa.int32()), ("pepmass", pa.float64()),
                         ("charge", pa.int8()), ("mz", pa.list_(pa.float32())), ("intensity", pa.list_(pa.float32()))])


def classify(proteins):
    """Class of each row: every accession, rev_ stripped, must carry the prefix, else 'ref'."""
    out = np.full(len(proteins), "ref", dtype=object)
    for name, prefix in [(x, x) for x in ("PSEUDO", "NNEG", "REJ", "PENTR", "MOD", "ENTR")] + [("CHIM", "CHIMPEP")]:
        pattern = rf"^(rev_)?{prefix}_[^;]*(;(rev_)?{prefix}_[^;]*)*$"
        out[pc.match_substring_regex(proteins, pattern).to_numpy(zero_copy_only=False)] = name
    return out


def keys_of(table):
    scans = table["scannr"].to_pandas().str.extract(r"scan=(\d+)", expand=False).astype(np.int64)
    return pd.DataFrame({"filename": table["filename"].to_pandas(), "scan": scans})


def anti_join(table, other):
    return table[~pd.MultiIndex.from_frame(table[KEY]).isin(pd.MultiIndex.from_frame(other[KEY]))]


def half_of(table):
    return table.filename.map(plex_half).values


def scan_spectra(parquet):
    novel, ranked = [], []
    for i in range(parquet.num_row_groups):
        t = parquet.read_row_group(i, columns=["filename", "scannr", "proteins", "rank"])
        keys = keys_of(t)
        novel.append(keys[classify(t["proteins"]) != "ref"])
        ranked.append(keys[t["rank"].to_numpy() == 1])
    return pd.concat(novel).drop_duplicates(), pd.concat(ranked).drop_duplicates()


def original_labels(cohort):
    """Rank-1 rows of the original search; an accession list is cut at 4,000 characters."""
    t = pq.read_table(f"results/cptac/pan/sage/{cohort}/results.sage.parquet", filters=[("rank", "=", 1)],
                      columns=["filename", "scannr", "peptide", "stripped_peptide", "proteins", "is_decoy",
                               "spectrum_q"])
    labels = keys_of(t).assign(orig_peptide=t["peptide"].to_pandas(), orig_stripped=t["stripped_peptide"].to_pandas(),
                               orig_decoy=t["is_decoy"].to_pandas(), orig_q=t["spectrum_q"].to_pandas(),
                               orig_acc=t["proteins"].to_pandas().str.slice(0, 4000))
    is_entr = np.where(labels.orig_acc.str.contains("ENTR_"), "ENTR", "ref")
    labels["orig_cls"] = np.where(labels.orig_acc.str.contains("CHIMPEP_"), "CHIM", is_entr)
    target = ~labels.orig_decoy
    labels["positive"] = (labels.orig_q <= CONFIG["training"]["label_q_pos"]) & target & (labels.orig_cls == "ref")
    labels["confident"] = (labels.orig_q <= OB["label_q"]) & target
    return labels


def label_fold_finder():
    """Returns folds(accessions): the folds of all accessions of a label, as the first alone misses paralogs; one cut at
    4,000 characters is ignored."""
    junctions = pd.read_csv(f"{OPTIONB_ROOT}/pool/junctions.tsv.gz", sep="\t", usecols=["name", "fold"])
    groups = pd.read_csv(f"{OPTIONB_ROOT}/pool/chrom_groups.tsv", sep="\t", dtype={"chr": str})
    fold_of = gene_chromosomes().map(groups.set_index("chr").chrom_group).dropna().astype(int).to_dict()
    fold_of.update(junctions.drop_duplicates("name").set_index("name").fold.to_dict())
    genes = accession_gene_map()

    def folds(proteins):
        accessions = proteins.split(";")[:-1] if len(proteins) >= 4000 else proteins.split(";")
        return {fold_of.get(gene_of_accession(accession, genes)) for accession in accessions} - {None}

    return folds


def held_out_peptides(fold):
    return set(pd.read_csv(f"{OPTIONB_ROOT}/fold{fold}/pseudo.tsv", sep="\t").peptide.map(il))


def write_roles(parquet, roles, labels, directory_of, union_path):
    """Second pass: all rows of the selected spectra with class and label columns, sorted by file, scan and rank."""
    union = pd.concat(roles.values())[KEY].drop_duplicates()
    wanted = pd.MultiIndex.from_frame(union)
    rows = []
    for i in range(parquet.num_row_groups):
        t = parquet.read_row_group(i)
        keys = keys_of(t)
        keep = pd.MultiIndex.from_frame(keys).isin(wanted)
        rows.append(t.filter(pa.array(keep)).append_column("scan", pa.array(keys.scan[keep])).drop(["scannr"]))
    table = pa.concat_tables(rows).to_pandas()
    table["cls"] = classify(pa.array(table.proteins.values))
    columns = KEY + ["orig_peptide", "orig_stripped", "orig_decoy", "orig_q", "orig_cls"]
    table = table.merge(labels[columns], on=KEY, how="left").sort_values(KEY + ["rank"]).reset_index(drop=True)
    for name, spectra in roles.items():
        os.makedirs(directory_of(name), exist_ok=True)
        table.merge(spectra, on=KEY).to_parquet(f"{directory_of(name)}/cands.parquet", compression="zstd", index=False)
    os.makedirs(os.path.dirname(union_path), exist_ok=True)
    union.to_parquet(union_path, index=False)


def select_fold(cohort, fold, n_ref=400_000, n_junk=250_000):
    """eval = all spectra with a novel-class row; train_h<h>: labelled novel spectra of half h, reference and junk."""
    out = f"{SELECTION_ROOT}/{cohort}_f{fold}"
    rng = np.random.default_rng(OB["seed"] + fold)
    parquet = pq.ParquetFile(f"{OPTIONB_ROOT}/sage/{cohort}_f{fold}/results.sage.parquet")
    novel, ranked = scan_spectra(parquet)
    labels = original_labels(cohort)
    folds_of = label_fold_finder()
    held_out = set().union(*[held_out_peptides(f) for f in range(OB["n_folds"]) if f != fold])

    def usable_label(table):
        in_group = [bool(fs) and all(x % 2 == fold % 2 for x in fs) for fs in map(folds_of, table.orig_acc)]
        return np.array(in_group) & ~table.orig_stripped.fillna("").map(il).isin(held_out).values

    def in_split(table):
        return np.array([crc(f"{f}:{x}") % 3 for f, x in zip(table.filename, table.scan)]) == fold // 2

    roles = {"eval": novel.assign(sel="novel")}
    for half in (0, 1):
        novel_half = novel[half_of(novel) == half]
        labels_half = labels[half_of(labels) == half]
        positive = labels_half[labels_half.positive]
        labelled_novel = novel_half.merge(positive, on=KEY)
        labelled_novel = labelled_novel[usable_label(labelled_novel)]
        reference = anti_join(positive, novel_half)
        reference = reference[in_split(reference)]
        reference = reference.iloc[rng.permutation(len(reference))[: int(n_ref * 1.6)]]
        reference = reference[usable_label(reference)].head(n_ref)
        spectra = ranked[half_of(ranked) == half]
        junk = anti_join(anti_join(spectra[in_split(spectra)], labels_half[labels_half.confident]), novel_half)
        junk = junk.iloc[rng.permutation(len(junk))[:n_junk]]
        role = pd.concat([labelled_novel[KEY].assign(sel="novel"), reference[KEY].assign(sel="ref"),
                          junk.assign(sel="junk")], ignore_index=True)
        roles[f"train_h{half}"] = role.drop_duplicates(KEY)
    write_roles(parquet, roles, labels, lambda name: f"{out}/{name}", f"{out}/union_keys.parquet")
    peaks, shard = f"{RESULTS}/spectra_v3/{cohort}_f{fold}/spectra.parquet", f"{SHARDS}/{cohort}_f{fold}_eval"
    extract(f"data/cptac_mgf/{cohort}", f"{out}/union_keys.parquet", peaks)
    build(f"{out}/eval/cands.parquet", peaks, shard, "eval")
    truncate(shard, f"{shard}_trunc_full")
    for half in (0, 1):
        build(f"{out}/train_h{half}/cands.parquet", peaks, f"{SHARDS}/{cohort}_f{fold}_train_h{half}", "train")
        split(shard, half)
        truncate(f"{shard}_h{half}", f"{shard}_h{half}_trunc")


def select_pretrain(cohort, n_ref=1_000_000, n_junk=300_000):
    """g<g>h<h>: reference spectra of half h labelled only in group g, not held out in group 1 - g, plus shared junk."""
    rng = np.random.default_rng(OB["seed"] + 77)
    parquet = pq.ParquetFile(f"results/cptac/pan/sage/{cohort}/results.sage.parquet")
    novel, ranked = scan_spectra(parquet)
    labels = original_labels(cohort)
    held_out = {g: set().union(*[held_out_peptides(f) for f in range(g, OB["n_folds"], 2)]) for g in (0, 1)}
    folds_of = label_fold_finder()
    reference = anti_join(labels[labels.positive], novel)
    groups = [{x % 2 for x in folds} for folds in map(folds_of, reference.orig_acc)]
    reference = reference.assign(g=[next(iter(g)) if len(g) == 1 else -1 for g in groups], h=half_of(reference),
                                 ilseq=reference.orig_stripped.fillna("").map(il))
    junk = anti_join(anti_join(ranked, labels[labels.confident]), novel)
    junk = junk.assign(h=half_of(junk))
    roles = {}
    for half in (0, 1):
        reference_half = reference[reference.h == half]
        reference_half = reference_half.iloc[rng.permutation(len(reference_half))]
        junk_half = junk[junk.h == half]
        junk_half = junk_half.iloc[rng.permutation(len(junk_half))[:n_junk]]
        for g in (0, 1):
            kept = reference_half[(reference_half.g == g) & ~reference_half.ilseq.isin(held_out[1 - g])].head(n_ref)
            roles[f"g{g}h{half}"] = pd.concat([kept[KEY].assign(sel="ref"), junk_half[KEY].assign(sel="junk")],
                                              ignore_index=True)
    keys = f"{SELECTION_ROOT}/{cohort}_pre/union_keys.parquet"
    peaks = f"{RESULTS}/spectra_v3/{cohort}_pre/spectra.parquet"
    write_roles(parquet, roles, labels, lambda name: f"{SELECTION_ROOT}/{cohort}_pre_{name}", keys)
    extract(f"data/cptac_mgf/{cohort}", keys, peaks)
    for name in roles:
        build(f"{SELECTION_ROOT}/{cohort}_pre_{name}/cands.parquet", peaks, f"{SHARDS}/{cohort}_pre_{name}", "train")


def extract(mgf_root, keys_path, out):
    """Peak table of the keys: the 200 most intense peaks outside the TMT reporter window, sorted by m/z."""
    keys = pq.read_table(keys_path, columns=KEY).to_pandas().drop_duplicates()
    scans_of = keys.groupby("filename").scan.agg(set).to_dict()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with pq.ParquetWriter(out, PEAK_SCHEMA, compression="zstd") as writer:
        for path in sorted(glob.glob(f"{mgf_root}/*/*.mgf")):
            filename = os.path.basename(path)
            rows = []
            for lines in mgf_blocks(path):
                scan = int(lines[0].rsplit("scan=", 1)[1])
                if scan in scans_of.get(filename, ()):
                    pepmass, charge, _, mz, intensity = parse_block(lines)
                    rows.append((filename, scan, pepmass, charge or 0,
                                 *reduce_peaks(mz, intensity, 200, sort_kind="stable")))
            table = pd.DataFrame(rows, columns=PEAK_SCHEMA.names)
            writer.write_table(pa.Table.from_pandas(table, schema=PEAK_SCHEMA, preserve_index=False))


def peak_batches(spectra, columns):
    for path in [spectra] if isinstance(spectra, str) else spectra:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=200_000, columns=columns):
            yield batch.to_pandas()


def peak_arrays(sp, spectra):
    """Top peaks of each spectrum, m/z ascending, and sqrt(I / I_max)."""
    row_of = dict(zip(zip(sp.filename, sp.scan.tolist()), range(len(sp))))
    mz = np.zeros((len(sp), N_PEAKS), np.float32)
    it = np.zeros((len(sp), N_PEAKS), np.float16)
    for batch in peak_batches(spectra, KEY + ["mz", "intensity"]):
        for key, m, x in zip(zip(batch.filename, batch.scan.tolist()), batch.mz, batch.intensity):
            if key in row_of:
                if len(m) > N_PEAKS:
                    order = np.sort(np.lexsort((m, -x))[:N_PEAKS])
                    m, x = m[order], x[order]
                mz[row_of[key], :len(m)] = m
                it[row_of[key], :len(x)] = np.sqrt(x / max(float(x.max(initial=0)), 1e-9))
    return mz, it


def build(cands, spectra, out, role):
    """Role train removes decoys, PENTR and REJ and labels at q <= 0.001; role eval keeps all rows, q <= 0.01."""
    K = K_CANDIDATES
    os.makedirs(out, exist_ok=True)
    df = pd.read_parquet(cands)
    if role == "train":
        df = df[df.cls.isin(["ref", "PSEUDO", "NNEG"]) & ~df.is_decoy]
    df = df.sort_values(KEY + ["rank"]).reset_index(drop=True)
    df["k"] = df.groupby(KEY).cumcount()
    df = df[df.k < K].reset_index(drop=True)
    df["i"] = df.groupby(KEY, sort=False).ngroup()
    sp = df.drop_duplicates(KEY).reset_index(drop=True)
    n, ii, kk = len(sp), df.i.values, df.k.values

    def slots(values, dtype, shape=()):
        array = np.zeros((n, K) + shape, dtype)
        array[ii, kk] = values
        return array

    features = df.assign(log_scored_candidates=np.log1p(df.scored_candidates.astype(float)),
                         abs_precursor_ppm=df.precursor_ppm.abs(), abs_delta_rt_model=df.delta_rt_model.abs(),
                         log_ms2_intensity=np.log1p(df.ms2_intensity.astype(float)))[FEATS].to_numpy(np.float32)
    tok = np.zeros((n, K, L_MAX), np.int8)
    token_id = {token: i for i, token in enumerate(TOKENS)}
    residues = df.peptide.str.replace(r"\[\+229\.16293\]-?|\[\+57\.021465\]", "", regex=True)
    for row, slot, peptide in zip(ii, kk, residues.str.replace("M[+15.994915]", "m", regex=False)):
        tok[row, slot, :len(peptide)] = [token_id[a] for a in peptide]
    # slot of the candidate equal to the original rank-1 reference target at q <= q_label, K if absent
    q_label = CONFIG["training"]["label_q_pos"] if role == "train" else CONFIG["optionb"]["label_q"]
    is_target = ~sp.orig_decoy.fillna(True).values.astype(bool)
    confident = (sp.orig_q.values <= q_label) & (sp.orig_cls.values == "ref") & is_target
    hit = df[df.peptide.values == df.i.map(sp.orig_peptide).values]
    first_slot = hit[~hit.is_decoy].groupby("i").k.min()
    label = np.full(n, -1, np.int8)
    label[confident] = K
    rows = first_slot.index.values
    label[rows[confident[rows]]] = first_slot.values[confident[rows]]
    if role == "train":
        label[(sp.sel.values == "junk") & ~confident] = K
    mz, it = peak_arrays(sp, spectra)
    arrays = dict(mz=mz, it=it, prec=np.c_[sp.expmass, sp.charge, sp.aligned_rt].astype(np.float32), tok=tok,
                  cmask=slots(True, bool), cdecoy=slots(df.is_decoy.values, bool),
                  ccls=slots(df.cls.map(CLASS_CODE).values, np.int8), cfeat=slots(features, np.float32, (len(FEATS),)),
                  label=label)
    for name, array in arrays.items():
        np.save(f"{out}/{name}.npy", array)
    groups = df.groupby("i")
    meta = sp[KEY + ["orig_q", "orig_stripped", "orig_cls"]]
    meta = meta.assign(cand_peptide=groups.peptide.apply(list).values,
                       cand_stripped=groups.stripped_peptide.apply(list).values)
    meta.to_parquet(f"{out}/meta.parquet", index=False)


def split(shard, half):
    os.makedirs(f"{shard}_h{half}", exist_ok=True)
    meta = pd.read_parquet(f"{shard}/meta.parquet")
    keep = np.flatnonzero(meta.filename.map(plex_half).values == half)
    for path in glob.glob(f"{shard}/*.npy"):
        np.save(f"{shard}_h{half}/{os.path.basename(path)}", np.load(path, mmap_mode="r")[keep])
    meta.iloc[keep].reset_index(drop=True).to_parquet(f"{shard}_h{half}/meta.parquet", index=False)


def truncate(shard, out):
    """Production-like twin: all rows if the rank-1 candidate (slot 0) is novel-class, else rank 1 + the novel rows."""
    os.makedirs(out, exist_ok=True)
    cm = np.load(f"{shard}/cmask.npy")
    novel = np.isin(np.load(f"{shard}/ccls.npy"), NOVEL_CODES) & cm
    for name in set(os.listdir(shard)) - {"cmask.npy"}:
        os.link(f"{shard}/{name}", f"{out}/{name}")
    np.save(f"{out}/cmask.npy", cm & (novel[:, [0]] | novel | (np.arange(cm.shape[1])[None, :] == 0)))


def stitch(cohort, fold):
    """Half scores back in the row order of the full eval shard."""
    meta = pd.read_parquet(f"{SHARDS}/{cohort}_f{fold}_eval/meta.parquet", columns=["filename"])
    half = meta.filename.map(plex_half).values
    for suffix in ("", "_trunc"):
        stitched = np.empty((len(half), K_CANDIDATES), np.float32)
        for h in (0, 1):
            run = f"{RESULTS}/runs/s1_g{1 - fold % 2}h{1 - h}"
            stitched[half == h] = np.load(f"{run}/score_v3_{cohort}_f{fold}_eval_h{h}{suffix}_fp32/scores.npy")
        os.makedirs(f"{RESULTS}/runs/stitched/v3_{cohort}_f{fold}{suffix}", exist_ok=True)
        np.save(f"{RESULTS}/runs/stitched/v3_{cohort}_f{fold}{suffix}/scores.npy", stitched)


def spectrum_info(spectra, keys):
    """pepmass, charge and Sage's total intensity of the key spectra: the top 150 peaks with m/z >= 150."""
    wanted = pd.MultiIndex.from_frame(keys)
    parts = []
    for batch in peak_batches(spectra, KEY + ["pepmass", "charge", "mz", "intensity"]):
        batch = batch[pd.MultiIndex.from_frame(batch[KEY].astype({"scan": np.int64})).isin(wanted)]
        total = []
        for mz, intensity in zip(batch.mz, batch.intensity):
            x = intensity.astype(np.float64)[mz >= 150.0]
            total.append((np.partition(x, len(x) - 150)[-150:] if len(x) > 150 else x).sum())
        parts.append(batch[KEY + ["pepmass"]].assign(mgf_charge=batch.charge, tot_int=total))
    return pd.concat(parts, ignore_index=True).astype({"scan": np.int64})


def align_rt(cc, min_points=50):
    """Sage-like global RT alignment of each file to the consensus of the confident rank-1 peptides in >= 2 files."""
    rt_n = cc.rt / cc.groupby("filename").rt.transform("max")
    confident = cc[(cc["rank"] == 1) & (cc.spectrum_q <= 0.01) & ~cc.is_decoy.astype(bool)].assign(rtn=rt_n)
    per_file = confident.groupby(["peptide", "filename"]).rtn.median().reset_index()
    n_files = per_file.groupby("peptide").size()
    consensus = per_file.groupby("peptide").rtn.median().rename("cons")
    points = per_file[per_file.peptide.map(n_files).values >= 2].join(consensus, on="peptide")
    fits = {f: (np.polyfit(g.rtn, g.cons, 1) if len(g) >= min_points else np.array([1.0, 0.0]))
            for f, g in points.groupby("filename")}
    slope = cc.filename.map(lambda f: fits.get(f, (1.0, 0.0))[0]).astype(float)
    intercept = cc.filename.map(lambda f: fits.get(f, (1.0, 0.0))[1]).astype(float)
    return slope * rt_n + intercept


def production(cohort, chunk=1_000_000):
    """Shards of class_cands: spectra of plex half h, sorted, in chunks, with the columns class_cands lacks."""
    if cohort in OPTIONB_COHORTS:
        spectra = f"{RESULTS}/prod/spectra/{cohort}/spectra.parquet"
        extract(f"data/cptac_mgf/{cohort}", f"results/cptac/pan/cohorts/{cohort}/class_cands.parquet", spectra)
    else:
        spectra = glob.glob(f"results/cptac/pan/spectra/{cohort}/chunks/*.parquet")
        spectra = sorted(spectra) or f"results/cptac/pan/spectra/{cohort}/spectra.parquet"
    cc = pd.read_parquet(f"results/cptac/pan/cohorts/{cohort}/class_cands.parquet")
    cc = cc.astype({"filename": str, "scan": np.int64})
    cc = cc.merge(spectrum_info(spectra, cc[KEY].drop_duplicates()), on=KEY, how="left")
    cc["delta_best"] = cc.groupby(KEY).hyperscore.transform("max") - cc.hyperscore
    cc["longest_y_pct"] = cc.longest_y / cc.peptide_len
    cc["expmass"] = (cc.pepmass - PROTON) * cc.mgf_charge
    cc["aligned_rt"] = align_rt(cc)
    cc["delta_rt_model"] = (cc.aligned_rt - cc.predicted_rt).abs()
    cc["ms2_intensity"] = cc.matched_intensity_pct / 100.0 * cc.tot_int
    cc = cc.drop(columns=["pepmass", "mgf_charge", "tot_int"])
    cc = cc.assign(orig_peptide=None, orig_stripped=None, orig_cls=None, orig_q=np.nan, orig_decoy=np.nan, sel="prod")
    half = cc.filename.map(plex_half)
    for h in (0, 1):
        d = cc[half == h]
        keys = d[KEY].drop_duplicates().sort_values(KEY).reset_index(drop=True)
        d = d.merge(keys.assign(c=keys.index // chunk), on=KEY)
        for c, rows in d.groupby("c"):
            name = f"{cohort}_h{h}_c{c}"
            os.makedirs(f"{RESULTS}/prod/cands/{name}", exist_ok=True)
            rows.drop(columns="c").to_parquet(f"{RESULTS}/prod/cands/{name}/cands.parquet", index=False)
            build(f"{RESULTS}/prod/cands/{name}/cands.parquet", spectra, f"{RESULTS}/prod/shards/{name}", "eval")


if __name__ == "__main__":
    mode, pdc, *fold_number = sys.argv[1:]
    {"fold": select_fold, "pretrain": select_pretrain, "production": production, "stitch": stitch}[mode](
        pdc, *map(int, fold_number))
