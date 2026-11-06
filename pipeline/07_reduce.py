"""Reduce one Sage top-5 table and its MGFs to what the analyses use, so that both can be deleted.
usage: 07_reduce.py <results.sage.parquet> <mgf_root> <out_dir> <reporter.parquet> <spectra_dir>"""
import glob
import gzip
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from juncta.spectra import mgf_blocks, parse_block, reduce_peaks

PSM_COLUMNS = ["filename", "scannr", "peptide", "stripped_peptide", "proteins", "num_proteins", "rank", "is_decoy",
               "charge", "calcmass", "peptide_len", "missed_cleavages", "isotope_error", "precursor_ppm",
               "fragment_ppm", "hyperscore", "delta_next", "rt", "predicted_rt", "matched_peaks", "longest_b",
               "longest_y", "matched_intensity_pct", "scored_candidates", "poisson", "sage_discriminant_score",
               "spectrum_q"]
LIGHT_COLUMNS = ["filename", "scannr", "stripped_peptide", "peptide", "proteins", "num_proteins", "rank", "is_decoy",
                 "charge", "calcmass", "hyperscore", "sage_discriminant_score", "spectrum_q", "peptide_q", "rt",
                 "predicted_rt", "matched_peaks"]
SPECTRA_SCHEMA = pa.schema([("filename", pa.string()), ("scan", pa.int32()), ("pepmass", pa.float64()),
                            ("charge", pa.int8()), ("rt", pa.float32()), ("mz", pa.list_(pa.float32())),
                            ("intensity", pa.list_(pa.float32()))])

results, mgf_root, out, reporter_file, spectra_dir = sys.argv[1:]
os.makedirs(out, exist_ok=True)
os.makedirs(spectra_dir, exist_ok=True)
source = pq.ParquetFile(results)
paths = {os.path.basename(p): p for p in glob.glob(f"{mgf_root}/*/*.mgf")}


def spectrum_keys(filename, scan):
    file_code = pd.Categorical(filename, categories=sorted(paths)).codes.astype(np.int64)
    return file_code * 10_000_000 + np.asarray(scan, np.int64)


def row_groups(columns):
    """Yield row groups with cls (CHIM / ENTR = every accession is CHIMPEP_* / ENTR_*, rev_ stripped), scan and key."""
    for i in range(source.num_row_groups):
        t = source.read_row_group(i, columns=[c for c in columns if c in source.schema_arrow.names]).to_pandas()
        n_accessions = t.proteins.str.count(";") + 1
        t["filename"] = t.filename.astype(str)
        t["cls"] = np.where(t.proteins.str.count("CHIMPEP_") == n_accessions, "CHIM",
                            np.where(t.proteins.str.count("ENTR_") == n_accessions, "ENTR", "ref"))
        t["scan"] = t.pop("scannr").str.extract(r"scan=(\d+)", expand=False).astype(np.int32)
        t["key"] = spectrum_keys(t.filename, t.scan)
        yield t


# Pass 1: spectra with CHIM/ENTR candidates, decoy pool, rank-1 reference targets at spectrum_q <= 0.05, per-file QC.
candidate_keys, rank1_candidate_keys, decoys, reference, qc = [], [], [], [], []
for t in row_groups(LIGHT_COLUMNS):
    candidate_keys.append(t.key[t.cls != "ref"].unique())
    t = t[t["rank"] == 1]
    rank1_candidate_keys.append(t.key[t.cls != "ref"].unique())
    qc.append(t.assign(t01=(t.spectrum_q <= 0.01) & ~t.is_decoy).groupby("filename").t01.agg(r1="size", t01="sum"))
    decoys.append(t[t.is_decoy].groupby("stripped_peptide").sage_discriminant_score.max())
    reference.append(t[(t.cls == "ref") & ~t.is_decoy & (t.spectrum_q <= 0.05)])
candidate_keys, rank1_candidate_keys = np.concatenate(candidate_keys), np.concatenate(rank1_candidate_keys)
pd.concat(decoys).groupby(level=0).max().rename("T").reset_index().to_parquet(f"{out}/decoy_best.parquet")
pd.concat(qc).groupby("filename").sum().to_csv(f"{out}/qc.tsv", sep="\t")
reference = pd.concat(reference, ignore_index=True)
reporter_keys = np.concatenate([candidate_keys, reference.key[reference.spectrum_q <= 0.01]])
reference = reference.drop(columns=["rank", "is_decoy", "cls", "key"])
reference["proteins"] = reference.proteins.str.split(";").str[:5].str.join(";")
reference.to_parquet(f"{out}/r1_ref.parquet", compression="zstd", index=False)

# Pass 2: rows of those spectra; reference rows keep their first accession.
kept = []
for t in row_groups(PSM_COLUMNS):
    t = t[t.key.isin(candidate_keys) & (t.key.isin(rank1_candidate_keys) | (t.cls != "ref") | (t["rank"] == 1))].copy()
    t["proteins"] = t.proteins.where(t.cls != "ref", t.proteins.str.split(";").str[0])
    kept.append(t.drop(columns="key"))
kept = pd.concat(kept, ignore_index=True)
kept.to_parquet(f"{out}/class_cands.parquet", compression="zstd", index=False)

# Spectra to keep: rank-1 CHIM/ENTR PSMs at spectrum_q <= 0.1 and 20,000 random confident reference PSMs.
columns = ["filename", "scan", "cls", "is_decoy", "stripped_peptide", "spectrum_q"]
candidates = kept[(kept["rank"] == 1) & (kept.cls != "ref") & (kept.spectrum_q <= 0.1)][columns]
sample = reference[reference.spectrum_q <= 0.001]
sample = sample.sample(n=min(20000, len(sample)), random_state=1)   # the row order of r1_ref is part of the draw
sample = sample.assign(cls="ref", is_decoy=False)[columns]
candidates = pd.concat([candidates.assign(why="chim_entr_r1_q0.1"), sample.assign(why="ref_baseline_q0.001")],
                       ignore_index=True).drop_duplicates(["filename", "scan"])
candidates.to_csv(f"{out}/cand_spectra.tsv", sep="\t", index=False)
candidate_scans = {f: set(g.scan) for f, g in candidates.groupby("filename")}
key_scans = {f: set(g.scan) for f, g in kept[["filename", "scan"]].drop_duplicates().groupby("filename")}


def extract(name):
    """Of one MGF: the candidate spectra retitled <filename>:<scan>, and the peaks of the class_cands spectra."""
    blocks, rows = [], []
    for lines in mgf_blocks(paths[name]):
        scan = int(lines[0].rsplit("scan=", 1)[1])
        if scan in candidate_scans.get(name, ()):
            blocks.append(f"BEGIN IONS\nTITLE={name}:{scan}\n" + "".join(lines[1:]) + "END IONS\n")
        if scan in key_scans.get(name, ()):
            pepmass, charge, rt, mz, it = parse_block(lines)
            rows.append((name, scan, pepmass, charge or 0, rt, *reduce_peaks(mz, it, 200, sort_kind="stable")))
    return blocks, rows


with (gzip.open(f"{out}/cand_spectra.mgf.gz", "wt", compresslevel=6) as mgf, Pool(16) as pool,
      pq.ParquetWriter(f"{spectra_dir}/spectra.parquet", SPECTRA_SCHEMA, compression="zstd") as store):
    for blocks, rows in pool.imap(extract, sorted(paths)):
        mgf.writelines(blocks)
        if rows:
            arrays = [pa.array(column, field.type) for column, field in zip(zip(*rows), SPECTRA_SCHEMA)]
            store.write_table(pa.table(arrays, schema=SPECTRA_SCHEMA))

reporters = pq.read_table(reporter_file)
wanted = np.isin(spectrum_keys(reporters["filename"].to_pandas().astype(str), reporters["scan"]), reporter_keys)
pq.write_table(reporters.filter(pa.array(wanted)), f"{out}/reporter_sel.parquet", compression="zstd")
