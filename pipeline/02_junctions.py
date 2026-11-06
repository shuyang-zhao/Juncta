"""Junction tables of every RNA source; each writes <out>/junctions_core.tsv.gz, the input of 03_orf.py.
usage: 02_junctions.py lalani <out> | recount3 <out> <tcga_dir> <gtex_dir> | controls <out> <tcga_dir> <catalog.tsv.gz>
Sources: Lalani et al. 2026 (doi:10.1126/sciadv.aei0901), cis-SAGe of recount3, canonical CDS introns as controls."""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from juncta.peptide import read_gtf
from juncta.stats import bh_adjust

COORDINATES = ["chr", "start", "end", "strand"]


def matrix_file(directory, kind):
    return glob.glob(f"{directory}/*.junctions.*.UNIQUE.{kind}.gz")[0]


def without_chr(chromosome):
    return chromosome.str.replace("^chr", "", regex=True).replace({"M": "MT"})


def write_core(out, t):
    """junctions_core.tsv.gz of one-strand introns; donor and acceptor are 0-based bases of the 1-based intron."""
    plus = t.strand == "+"
    t = t.assign(chr_d=t.chr, chr_a=t.chr, str_d=t.strand, str_a=t.strand, pos_d=np.where(plus, t.start - 2, t.end),
                 pos_a=np.where(plus, t.end, t.start - 2), tier="A")
    t.drop(columns=COORDINATES).to_csv(f"{out}/junctions_core.tsv.gz", sep="\t", index=False)


def lalani(out):
    """Tier1, high-confidence growth promoters and listed (non-)essential CCLE chimeric RNAs."""
    source = "data/lalani2026"
    calls = pd.read_csv(f"{source}/SupplementalTable1.tsv.gz", sep="\t", low_memory=False)
    calls["tier_rank"] = calls.Confidence.map({"Tier1": 1, "Tier2": 2, "Tier3": 3, "Tier4": 4})
    key = ["LeftChr", "LeftPosition", "LeftStrand", "RightChr", "RightPosition", "RightStrand"]
    junctions = calls.groupby(key).agg(FusionName=("FusionName", "first"), FusionID=("FusionID", "first"),
                                       Class=("Class", "first"), n_lines=("Cell_Line", "nunique"),
                                       tier=("tier_rank", "min"),
                                       inWGS=("inWGS", lambda v: ";".join(sorted(set(v.dropna()))))).reset_index()
    site = ["LeftChr", "LeftPosition", "RightChr", "RightPosition"]
    screen = pd.read_csv(f"{source}/SupplementalTable3.tsv.gz", sep="\t", low_memory=False,
                         usecols=["FusionName", *site, "growth_promoter", "high_conf_growth_promoter"])
    screen = screen.groupby(site).agg(gp=("growth_promoter", "max"), hc=("high_conf_growth_promoter", "max"))
    junctions = junctions.merge(screen.reset_index(), on=site, how="left")
    junctions["dep"] = np.select([junctions.hc.eq(True), junctions.gp.eq(True)], ["high_conf", "candidate"], "none")
    essential = pd.concat([pd.read_csv(f"{source}/SupplementalTable{k}.tsv.gz", sep="\t")[["FusionName", "Condition"]]
                           for k in (4, 5)]).drop_duplicates("FusionName")
    junctions = junctions.merge(essential, on="FusionName", how="left").rename(columns={"Condition": "essentiality"})
    junctions["essentiality"] = junctions.essentiality.fillna("none")
    junctions = junctions[(junctions.tier == 1) | (junctions.dep == "high_conf")
                          | (junctions.essentiality != "none")].copy()
    junctions["P_class"] = junctions.inWGS.str.contains("Fusion")
    ids = junctions.FusionID.str.split("-", n=1, expand=True)
    junctions["gene_d"], junctions["gene_a"] = ids[0].str.split(".").str[0], ids[1].str.split(".").str[0]
    junctions["junction_id"] = [f"LAL{i:06d}" for i in range(1, len(junctions) + 1)]
    junctions.to_csv(f"{out}/lalani_junctions.tsv.gz", sep="\t", index=False)
    pd.DataFrame({"junction_id": junctions.junction_id, "chr_d": without_chr(junctions.LeftChr),
                  "pos_d": junctions.LeftPosition - 1, "str_d": junctions.LeftStrand,
                  "chr_a": without_chr(junctions.RightChr), "pos_a": junctions.RightPosition - 1,
                  "str_a": junctions.RightStrand, "gene_d": junctions.gene_d, "gene_a": junctions.gene_a,
                  "name_d": junctions.FusionName, "name_a": junctions.FusionName,
                  "tier": "A"}).to_csv(f"{out}/junctions_core.tsv.gz", sep="\t", index=False)


def read_matrix(directory, keep_rows):
    """Sample ids, junction reads per 1-based matrix column, and the entries of keep_rows, one table per chunk."""
    ids = pd.read_csv(matrix_file(directory, "ID"), sep="\t").rail_id.values
    total = np.zeros(len(ids) + 1)
    kept = []
    # The first line after the comments gives the matrix size.
    for chunk in pd.read_csv(matrix_file(directory, "MM"), sep=r"\s+", comment="%", header=0, names=["r", "c", "v"],
                             chunksize=20_000_000, dtype=np.int64):
        total += np.bincount(chunk.c.values, weights=chunk.v.values, minlength=len(ids) + 1)
        kept.append(chunk[chunk.r.isin(keep_rows)])
    return ids, total, kept


def read_junction_rows(directory):
    table = pd.read_csv(matrix_file(directory, "RR"), sep="\t", dtype={"chromosome": str})
    table["row"] = np.arange(1, len(table) + 1)
    table["chr"] = without_chr(table.chromosome)
    table = table[table.strand.isin(["+", "-"])]
    donor_base = np.where(table.strand == "+", table.start, table.end).astype(str)
    return table.assign(dkey=table.chr + table.strand + donor_base)


def support(catalog, directory, groups):
    """Per junction and sample group (None = all): positive samples, reads, mean CPM, mean donor share (>= 10 reads)."""
    table = read_junction_rows(directory)
    rows = catalog[COORDINATES].merge(table[[*COORDINATES, "row"]], on=COORDINATES, how="left").row
    rows = rows.fillna(0).astype(int)
    donor_rows = table[table.dkey.isin(set(catalog.dkey))]
    ids, total, kept = read_matrix(directory, set(rows) | set(donor_rows.row))
    matrix = pd.concat(kept)
    counts = matrix.assign(rail_id=ids[matrix.c.values - 1]).pivot_table(index="r", columns="rail_id", values="v",
                                                                         aggfunc="sum", fill_value=0)
    counts = counts.reindex(columns=ids, fill_value=0)
    donor_sum = counts.reindex(donor_rows.row).fillna(0).assign(dkey=donor_rows.dkey.values).groupby("dkey").sum()
    stats = pd.DataFrame(index=catalog.index)
    for name, group in groups.items():
        mask = np.ones(len(ids), bool) if group is None else np.isin(ids, group)
        reads = counts.reindex(rows.values).fillna(0)[ids[mask]].values
        donors = donor_sum.reindex(catalog.dkey.values).fillna(0)[ids[mask]].values
        stats[f"n_{name}"] = mask.sum()
        stats[f"pos_{name}"] = (reads >= 2).sum(1)
        stats[f"frac_{name}"] = stats[f"pos_{name}"] / mask.sum()
        stats[f"reads_{name}"] = reads.sum(1)
        stats[f"cpm_{name}"] = (reads / total[1:][mask] * 1e6).mean(1)
        stats[f"share_{name}"] = np.nanmean(np.where(donors >= 10, reads / np.maximum(donors, 1), np.nan), 1)
    return stats


def recount3(out, tumor_dir, gtex_dir):
    """cis-SAGe catalog: junctions between exons of two genes of one strand, < 200 kb; third-gene introns are R0."""
    gtf = read_gtf(["gene", "exon"], "gene_id", "gene_name", "gene_biotype", "transcript_id")
    genes = gtf[gtf.feat == "gene"].set_index("gene_id")
    exons = gtf[gtf.feat == "exon"]
    plus = exons.strand == "+"
    sites = exons.assign(donor=np.where(plus, exons.end, exons.start), acceptor=np.where(plus, exons.start, exons.end))
    donors = sites.groupby(["chr", "strand", "donor"]).gene_id.agg(set).to_dict()
    acceptors = sites.groupby(["chr", "strand", "acceptor"]).gene_id.agg(set).to_dict()
    exons = exons.sort_values(["transcript_id", "start"])
    annotated = exons.assign(first=exons.end + 1, last=exons.groupby("transcript_id").start.shift(-1) - 1)
    introns = annotated.dropna(subset="last").groupby(["chr", "first", "last"]).gene_id.agg(set).to_dict()

    def ranked(gene_ids):
        return sorted(gene_ids, key=lambda x: (genes.gene_biotype[x] != "protein_coding", genes.start[x], x))

    junction_table = read_junction_rows(tumor_dir)
    candidates = []
    for r in junction_table[junction_table.length < 200_000].itertuples():
        before, after = (r.start - 1, r.end + 1) if r.strand == "+" else (r.end + 1, r.start - 1)
        gene_d, gene_a = donors.get((r.chr, r.strand, before)), acceptors.get((r.chr, r.strand, after))
        if not gene_d or not gene_a:
            continue
        both = gene_d & gene_a   # genes owning both sites: read-through models or plain intragenic splicing
        gene_d, gene_a = gene_d - both, gene_a - both
        intron_genes = introns.get((r.chr, r.start, r.end), set())
        pairs = [(a, b) for a in ranked(gene_d) for b in ranked(gene_a)
                 if (genes.start[a] < genes.start[b] if r.strand == "+" else genes.end[a] > genes.end[b])]
        if pairs and not intron_genes & (gene_d | gene_a):
            a, b = pairs[0]
            candidates.append(dict(chr=r.chr, start=r.start, end=r.end, strand=r.strand, dkey=r.dkey, gene_d=a,
                                   gene_a=b, name_d=genes.gene_name[a], name_a=genes.gene_name[b],
                                   biotype_d=genes.gene_biotype[a], biotype_a=genes.gene_biotype[b],
                                   third_gene=bool(intron_genes | both)))
    catalog = pd.DataFrame(candidates)
    catalog = catalog[~(catalog.biotype_d.str.match("IG_|TR_") | catalog.biotype_a.str.match("IG_|TR_"))]   # V(D)J
    sample_type = pd.read_csv(glob.glob(f"{tumor_dir}/*.tcga.*.MD.gz")[0], sep="\t", low_memory=False)
    sample_type = sample_type.set_index("rail_id").cgc_sample_sample_type
    tumour_groups = {"T": sample_type.index[sample_type == "Primary Tumor"],
                     "Ntcga": sample_type.index[sample_type == "Solid Tissue Normal"]}
    catalog = pd.concat([catalog, support(catalog, tumor_dir, tumour_groups),
                         support(catalog, gtex_dir, {"Ngtex": None})], axis=1)
    catalog = catalog[(catalog.pos_T + catalog.pos_Ntcga + catalog.pos_Ngtex) >= 2].copy()
    p = [fisher_exact([[a, n - a], [b, m - b]], alternative="greater")[1]
         for a, n, b, m in zip(catalog.pos_T, catalog.n_T, catalog.pos_Ntcga, catalog.n_Ntcga)]
    pseudogene = catalog.biotype_d.str.contains("pseudogene") | catalog.biotype_a.str.contains("pseudogene")
    mhc = (catalog.chr == "6") & (catalog.end >= 28_510_120) & (catalog.start <= 33_480_577)   # GRCh38
    catalog["tumor_enriched"] = ((catalog.frac_T >= 0.05) & (bh_adjust(p) < 0.01)
                                 & (catalog.frac_T >= 3 * (catalog.frac_Ntcga + 0.01))
                                 & (catalog.frac_T >= 3 * (catalog.frac_Ngtex + 0.01))
                                 & ~(catalog.share_T < 2 * catalog.share_Ntcga) & ~pseudogene & ~mhc)
    catalog["cls"] = np.where(catalog.third_gene, "R0", "C1")
    catalog = catalog.sort_values(["tumor_enriched", "pos_T"], ascending=False)
    catalog["junction_id"] = [f"R3J{i:06d}" for i in range(1, len(catalog) + 1)]
    catalog.drop(columns="dkey").to_csv(f"{out}/cis_sage_catalog.tsv.gz", sep="\t", index=False)
    write_core(out, catalog[["junction_id", *COORDINATES, "gene_d", "gene_a", "name_d", "name_a"]])


def controls(out, tumor_dir, catalog_file):
    """Canonical CDS introns on the same cpm_T / frac_T scale, sampled in 0.25-log10 bins of cpm_T (300 per bin)."""
    gtf = read_gtf(["exon", "CDS"], "transcript_id", "gene_id", "gene_name", "transcript_biotype")
    gtf = gtf[gtf.transcript_biotype == "protein_coding"]
    cds = gtf[gtf.feat == "CDS"].groupby("transcript_id").agg(cs=("start", "min"), ce=("end", "max"))
    exons = gtf[gtf.feat == "exon"].join(cds, on="transcript_id").dropna(subset=["cs"])
    exons = exons.sort_values(["transcript_id", "start"]).assign(in_cds=lambda e: (e.end >= e.cs) & (e.start <= e.ce))
    following = exons.groupby("transcript_id")[["start", "in_cds"]].shift(-1)
    first = exons.in_cds & following.in_cds.eq(True)
    introns = exons[first].assign(start=exons.end[first] + 1, end=following.start[first].astype(int) - 1)
    introns = introns.rename(columns={"gene_name": "name"})[[*COORDINATES, "gene_id", "name"]]
    introns = introns.drop_duplicates(COORDINATES)
    canonical = read_junction_rows(tumor_dir)[[*COORDINATES, "row"]].merge(introns, on=COORDINATES)

    ids, total, kept = read_matrix(tumor_dir, set(canonical.row))
    metadata = pd.read_csv(glob.glob(f"{tumor_dir}/*.tcga.*.MD.gz")[0], sep="\t", low_memory=False,
                           usecols=["rail_id", "cgc_sample_sample_type"])
    tumor = metadata.set_index("rail_id").cgc_sample_sample_type.reindex(ids).eq("Primary Tumor").values
    tumor_by_column = np.concatenate([[False], tumor])   # matrix columns are 1-based
    sums = pd.DataFrame({"cpm": [], "pos": []})
    for k in kept:
        k = k[tumor_by_column[k.c.values]]
        sums = sums.add(pd.DataFrame({"r": k.r.values, "cpm": k.v.values / total[k.c.values] * 1e6,
                                      "pos": k.v.values >= 2}).groupby("r").sum(), fill_value=0)
    canonical["cpm_T"] = canonical.row.map(sums.cpm).fillna(0) / tumor.sum()
    canonical["frac_T"] = canonical.row.map(sums.pos).fillna(0) / tumor.sum()
    canonical.drop(columns="row").to_csv(f"{out}/canonical_all.tsv.gz", sep="\t", index=False)

    catalog = pd.read_csv(catalog_file, sep="\t", usecols=["cpm_T"])
    lo, hi = np.log10(catalog.cpm_T[catalog.cpm_T > 0].min()), np.log10(catalog.cpm_T.max())
    canonical = canonical[canonical.cpm_T > 0].copy()
    canonical["lb"] = np.floor(np.log10(canonical.cpm_T) / 0.25) * 0.25
    in_range = (canonical.lb >= np.floor(lo / 0.25) * 0.25 - 0.25) & (canonical.lb <= hi + 0.25)
    sampled = canonical[in_range].sample(frac=1, random_state=1).groupby("lb").head(300)
    write_core(out, sampled.rename(columns={"gene_id": "gene_d", "name": "name_d"}).assign(
        junction_id=[f"CTRL{i:06d}" for i in range(len(sampled))], gene_a=sampled.gene_id,
        name_a=sampled["name"]).drop(columns=["row", "lb"]))


mode, out_dir = sys.argv[1:3]
os.makedirs(out_dir, exist_ok=True)
{"lalani": lalani, "recount3": recount3, "controls": controls}[mode](out_dir, *sys.argv[3:])
