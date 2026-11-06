"""Chimeric ORFs and junction peptides of a junction table. usage: 03_orf.py <junctions_core.tsv.gz> <out_dir> [keep]
The chimeric transcript is translated from the annotated start of the donor, or of the acceptor when the donor is not
coding, and classed by frame; peptides found (I = L) in a reference are dropped, or flagged with keep (controls)."""
import os
import re
import sys
from functools import cache
from itertools import product

import numpy as np
import pandas as pd
import pysam

from juncta.peptide import cleavage_sites, found_in, matches, read_gtf

ENSEMBL_PEP = "data/ensembl91_pep/Homo_sapiens.GRCh38.pep.all.fa.gz"
REFERENCES = [ENSEMBL_PEP, "data/uniprot_ref/UP000005640_9606.fasta.gz",
              "data/uniprot_ref/UP000005640_9606_additional.fasta.gz", "data/dk/gencode.v50.pc_translations.fa.gz"]
RANK = ["complete", "ccds", "basic", "tsl1", "cds_len", "tx_len"]
NOT_CHIMERIC = ["stop_before_junction", "5utr_swap", "5utr_swap_noATG", "noncoding_partner"]
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")
CODON = {"".join(c): aa for c, aa in zip(product("TCAG", repeat=3),
                                         "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")}
OUT = sys.argv[2].rstrip("/") + "/"
keep_known = len(sys.argv) > 3
os.makedirs(OUT, exist_ok=True)
junctions = pd.read_csv(sys.argv[1], sep="\t", dtype={"chr_d": str, "chr_a": str})

gtf = read_gtf(["transcript", "exon", "CDS", "start_codon", "stop_codon"], "transcript_id", "gene_id")
gtf = gtf.rename(columns={"transcript_id": "tid"})
gtf["start"] -= 1
exons = gtf[gtf.feat == "exon"].sort_values(["tid", "start"])
transcripts = exons.groupby("tid").agg(gene_id=("gene_id", "first"), chr=("chr", "first"), strand=("strand", "first"),
                                       es=("start", list), ee=("end", list))
transcripts["tx_len"] = exons.assign(length=exons.end - exons.start).groupby("tid").length.sum()
attributes = gtf[gtf.feat == "transcript"].set_index("tid").attr
for flag, pattern in {"ccds": 'tag "CCDS"', "basic": 'tag "basic"', "tsl1": 'transcript_support_level "1',
                      "nf": "cds_start_NF|cds_end_NF"}.items():
    transcripts[flag] = attributes.str.contains(pattern)
transcripts = transcripts.join(gtf[gtf.feat.isin(["CDS", "stop_codon"])].groupby("tid").agg(cs=("start", "min"),
                                                                                            ce=("end", "max")))
codons = gtf[gtf.feat.isin(["start_codon", "stop_codon"])].groupby(["tid", "feat"]).size().unstack(fill_value=0)
both_codons = codons.index[(codons.start_codon > 0) & (codons.stop_codon > 0)]
transcripts["complete"] = transcripts.cs.notna() & transcripts.index.isin(both_codons) & ~transcripts.nf
for c in ["es", "ee"]:   # exons in transcript direction
    transcripts[c] = [x if strand == "+" else x[::-1] for x, strand in zip(transcripts[c], transcripts.strand)]


def tx_coordinate(t, p):
    """Genomic base -> transcript coordinate (None if not exonic)."""
    offset = 0
    for a, b in zip(t.es, t.ee):
        if a <= p < b:
            return offset + (p - a if t.strand == "+" else b - 1 - p)
        offset += b - a


transcripts["cds_s"] = [tx_coordinate(t, int(t.cs if t.strand == "+" else t.ce - 1)) if t.complete else np.nan
                        for t in transcripts.itertuples()]
transcripts["cds_e"] = [tx_coordinate(t, int(t.ce - 1 if t.strand == "+" else t.cs)) if t.complete else np.nan
                        for t in transcripts.itertuples()]
transcripts["cds_len"] = (transcripts.cds_e - transcripts.cds_s + 1).fillna(0)
TX = {t.Index: t for t in transcripts.itertuples()}
spans = exons[["gene_id", "tid", "start", "end"]].assign(
    acceptor=np.where(exons.strand == "+", exons.start, exons.end - 1),
    donor=np.where(exons.strand == "+", exons.end - 1, exons.start))


def choose(gene, pos, side):
    """Best transcript per (gene, position), first those whose exon ends (donor) or starts (acceptor) there."""
    c = pd.DataFrame({"gene_id": gene, "pos": pos}).drop_duplicates().merge(spans, on="gene_id")
    c = c[(c.pos >= c.start) & (c.pos < c.end)].join(transcripts[RANK], on="tid")
    c["exact"] = c.pos == c[side]
    c = c.sort_values(["gene_id", "pos", "exact"] + RANK + ["tid"],
                      ascending=[True, True, False] + [False] * len(RANK) + [True])
    return c.drop_duplicates(["gene_id", "pos"]).set_index(["gene_id", "pos"]).tid


junctions = junctions.join(choose(junctions.gene_d, junctions.pos_d, "donor").rename("tx_d"), on=["gene_d", "pos_d"])
junctions = junctions.join(choose(junctions.gene_a, junctions.pos_a, "acceptor").rename("tx_a"), on=["gene_a", "pos_a"])
# Default transcript per gene: same ranking without a position; the parental protein when the chosen one is non-coding.
default_tx = transcripts.reset_index().sort_values(["gene_id"] + RANK + ["tid"],
                                                   ascending=[True] + [False] * len(RANK) + [True])
default_tx = default_tx.drop_duplicates("gene_id").set_index("gene_id").tid
genome = pysam.FastaFile("data/ref/hg38_sequins_SIRV_ERCCs_longSIRVs.fa")


def genomic(c, strand, p, i, j):
    """Sense sequence at transcript-direction offsets [i, j) from genomic base p."""
    if strand == "+":
        return genome.fetch(c, max(p + i, 0), max(p + j, 0)).upper()
    return genome.fetch(c, max(p - j + 1, 0), max(p - i + 1, 0)).upper().translate(COMPLEMENT)[::-1]


@cache
def tx_sequence(tid):
    t = TX[tid]
    x = "".join(genome.fetch(t.chr, a, b).upper() for a, b in sorted(zip(t.es, t.ee)))
    return x if t.strand == "+" else x.translate(COMPLEMENT)[::-1]


def translate(x, start):
    """From start to the first stop; returns (protein, stop found)."""
    protein = "".join(CODON.get(x[i:i + 3], "X") for i in range(start, len(x) - 2, 3))
    return protein.split("*")[0], "*" in protein


def region(tid, c):
    if not isinstance(tid, str):
        return "nonexonic"
    t = TX[tid]
    if not t.complete:
        return "noncoding"
    return "5utr" if c < t.cds_s else ("cds" if c <= t.cds_e else "3utr")


result, chimeric = [], {}
for j in junctions.itertuples():
    cd = tx_coordinate(TX[j.tx_d], j.pos_d) if isinstance(j.tx_d, str) else None
    ca = tx_coordinate(TX[j.tx_a], j.pos_a) if isinstance(j.tx_a, str) else None
    region_d, region_a = region(j.tx_d, cd), region(j.tx_a, ca)
    five = tx_sequence(j.tx_d)[:cd + 1] if cd is not None else genomic(j.chr_d, j.str_d, j.pos_d, -199, 1)
    three = tx_sequence(j.tx_a)[ca:] if ca is not None else genomic(j.chr_a, j.str_a, j.pos_a, 0, 3000)
    x, junction = five + three, len(five)
    start, protein, stop = None, "", False
    if region_d == "cds":
        start = int(TX[j.tx_d].cds_s)
        protein, stop = translate(x, start)
        if stop and len(protein) < (junction - start) // 3:
            orf_class = "stop_before_junction"
        elif region_a in ("5utr", "cds"):
            acceptor = TX[j.tx_a]
            if (junction - start) % 3 == (ca - acceptor.cds_s) % 3:
                reaches_stop = stop and ca + (start + 3 * len(protein) + 2 - junction) == acceptor.cds_e
                orf_class = "in_frame" if reaches_stop else "in_frame_early_stop"
            else:
                orf_class = "frameshift"
        else:
            orf_class = "cds_to_3utr" if region_a == "3utr" else "cds_to_noncoding"
    elif region_d == "3utr":
        orf_class, start = "stop_before_junction", int(TX[j.tx_d].cds_s)
        protein, stop = translate(x, start)
    elif region_a == "5utr":
        orf_class, start = "5utr_swap", junction + int(TX[j.tx_a].cds_s) - ca
        protein, stop = translate(x, start)
    elif region_a == "cds":
        start = x.find("ATG", junction)
        if start >= 0:
            protein, stop = translate(x, start)
            in_frame = (ca + start - junction - TX[j.tx_a].cds_s) % 3 == 0
            orf_class = "5utr_swap_Ntrunc" if in_frame else "5utr_swap_offframe_ATG"
        else:
            orf_class, start = "5utr_swap_noATG", None
    else:
        orf_class = "noncoding_partner"
    if protein:
        chimeric[j.junction_id] = protein
    result.append(dict(junction_id=j.junction_id, tx_d=j.tx_d, tx_a=j.tx_a, txpos_d=cd, txpos_a=ca,
                       jx_src=["genomic", "partial", "transcript"][(cd is not None) + (ca is not None)],
                       d_region=region_d, a_region=region_a, orf_class=orf_class, orf_start=start, junction_nt=junction,
                       prot_len=len(protein), orf_stop=stop,
                       junction_aa=(junction - start) // 3 if start is not None and start < junction else -1))
orfs = pd.DataFrame(result)
orfs.to_csv(OUT + "junction_orf.tsv.gz", sep="\t", index=False)


complete = set(transcripts.index[transcripts.complete])
proteins = orfs.merge(junctions[["junction_id", "gene_d", "gene_a", "name_d", "name_a", "tier"]], on="junction_id")
proteins["prot_chim"] = proteins.junction_id.map(chimeric).fillna("")
for side in "da":
    proteins[f"ptx_{side}"] = [t if t in complete else default_tx.get(gene)
                               for t, gene in zip(proteins[f"tx_{side}"], proteins[f"gene_{side}"])]
    proteins[f"prot_{side}"] = [translate(tx_sequence(t), int(TX[t].cds_s))[0] if t in complete else ""
                                for t in proteins[f"ptx_{side}"]]
proteins[["junction_id", "orf_class", "tier", "name_d", "name_a", "gene_d", "gene_a", "ptx_d", "ptx_a", "junction_aa",
          "prot_chim", "prot_d", "prot_a"]].to_csv(OUT + "proteins.tsv.gz", sep="\t", index=False)


peptides = []
for r in proteins[~proteins.orf_class.isin(NOT_CHIMERIC) & (proteins.prot_chim.str.len() > 0)].itertuples():
    start, junction, junction_aa, p = int(r.orf_start), int(r.junction_nt), int(r.junction_aa), r.prot_chim
    cut = [0] + cleavage_sites(p) + [len(p)]
    # Trypsin, <= 2 missed cleavages, 7-35 aa.
    for a, b in [(a, b) for i, a in enumerate(cut[:-1]) for b in cut[i + 1:i + 4] if 7 <= b - a <= 35]:
        if r.orf_class in ("in_frame", "in_frame_early_stop"):
            keep, novelty = start + 3 * a <= junction - 3 and start + 3 * b >= junction + 3, "junction_spanning"
        elif r.orf_class == "5utr_swap_Ntrunc":
            keep, novelty = a == 0, "novel_Nterm"
        elif r.orf_class == "5utr_swap_offframe_ATG":
            keep, novelty = True, "novel_sequence"
        else:
            keep, novelty = b > junction_aa, "novel_sequence"
        if keep:
            peptides.append((p[a:b], r.junction_id, "orf", r.orf_class, r.tier, novelty))
pep = pd.DataFrame(peptides, columns=["peptide", "junction_id", "frame_source", "class", "tier", "novelty"])
pep = pep.drop_duplicates()
pep["in_reference"] = pep.peptide.isin(found_in(pep.peptide.unique(), REFERENCES))
pep = pep if keep_known else pep[~pep.in_reference]
pep = pep.merge(junctions[["junction_id", "name_d", "name_a"]], on="junction_id")
pep.to_csv(OUT + "junction_peptides.tsv", sep="\t", index=False)

if keep_known:   # in-frame reference peptides whose Ensembl proteins all come from transcripts with the junction
    controls = pep[(pep["class"] == "in_frame") & pep.in_reference]
    plus = junctions.str_d == "+"
    first_base = np.where(plus, junctions.pos_d, junctions.pos_a) + 2
    last_base = np.where(plus, junctions.pos_a, junctions.pos_d)
    intron_of = dict(zip(junctions.junction_id, zip(junctions.chr_d, first_base, last_base)))
    introns = exons.assign(first=exons.end + 1, last=exons.groupby("tid").start.shift(-1)).dropna(subset="last")
    tx_of_intron = introns.groupby(["chr", "first", "last"]).tid.agg(set).to_dict()
    tx_of_peptide = pd.DataFrame([(p, re.search(r"transcript:(\S+)", header).group(1).split(".")[0]) for header, p
                                  in matches(controls.peptide.unique(), [ENSEMBL_PEP])], columns=["peptide", "tid"])
    tx_of_peptide = tx_of_peptide.groupby("peptide").tid.agg(set).to_dict()
    specific = {p for p, j in zip(controls.peptide, controls.junction_id)
                if p in tx_of_peptide and tx_of_peptide[p] <= tx_of_intron.get(intron_of[j], set())}
    pd.Series(sorted(specific)).to_csv(OUT + "control_peptides_unique.txt", index=False, header=False)
