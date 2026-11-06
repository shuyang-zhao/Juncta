"""Union peptide database of all cohorts and its hypothesis flags: every cohort is searched against the same hypotheses.
usage: 04_union_db.py <out_dir>. Writes db.fasta (reference, CHIMPEP_<i>|<pep>, ENTR_<i>|<target>), peptide_groups.tsv,
chim_peptides_kept.txt, entrapment.tsv (one shuffle per kept peptide that digests like it), chimpep_flags.tsv."""
import os
import sys

import pandas as pd

from juncta.config import COHORTS
from juncta.peptide import REFERENCE, found_in, read_gtf, shuffle_entrapments

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
# Peptides of a junction with such a partner are dropped unless published.
FAMILY_FILTER = r"^(?:HLA-|IG[HKL][VJC]|TR[ABDG][VJC]|HB[ABDEGQZM]\d?$|MT1|MT2A$|SAA\d|TUBA|TUBB)"
COLUMNS = ["peptide", "junction_id", "source", "orf_class", "jclass", "name_d", "name_a", "biotype_d", "biotype_a",
           "lal_tier", "lal_dep", "lal_ess", "lal_P", "lal_n_lines", "jkey", "r3_project", "r3_frac_T", "r3_cpm_T",
           "r3_frac_Ntcga", "r3_frac_Ngtex", "r3_tumor_enriched", "novelty", "pub_study"]
PROJECT_DIRS = {"KIRC": "results/cptac/recount3_kirc",
                **{p: f"results/cptac/pan/r3/{p}" for p in sorted(v[1] for v in COHORTS.values()) if p != "KIRC"}}


genes = read_gtf(["gene"], "gene_id", "gene_name", "gene_biotype").set_index("gene_id")
drs = pd.read_csv("results/cptac/pepquery/input_drs_tierAB/peptide_map.tsv", sep="\t", dtype=str)
master = pd.read_csv("results/master/chrna_master.tsv.gz", sep="\t", dtype=str,
                     usecols=["junction_id", "biotype_d", "biotype_a", "category"])
drs = drs.drop(columns=["category", "novelty"]).merge(master, on="junction_id", how="left")
rows = [drs.rename(columns={"category": "jclass"}).assign(source="DRS")]

lalani = pd.read_csv("results/cptac/lalani/orf/junction_peptides.tsv", sep="\t")
lalani = lalani[lalani.frame_source == "orf"][["peptide", "junction_id", "class"]]
lalani = lalani.merge(pd.read_csv("results/cptac/lalani/lalani_junctions.tsv.gz", sep="\t"), on="junction_id")
rows.append(lalani.rename(columns={"class": "orf_class", "Class": "jclass", "tier": "lal_tier", "dep": "lal_dep",
                                   "essentiality": "lal_ess", "P_class": "lal_P", "n_lines": "lal_n_lines"}).assign(
    source="LALANI", name_d=lalani.gene_d.map(genes.gene_name), name_a=lalani.gene_a.map(genes.gene_name),
    biotype_d=lalani.gene_d.map(genes.gene_biotype), biotype_a=lalani.gene_a.map(genes.gene_biotype)))

for project, d in PROJECT_DIRS.items():
    catalog = pd.read_csv(f"{d}/cis_sage_catalog.tsv.gz", sep="\t", dtype={"chr": str})
    catalog["jkey"] = catalog.chr + ":" + catalog.start.astype(str) + "-" + catalog.end.astype(str) + catalog.strand
    jp = pd.read_csv(f"{d}/orf/junction_peptides.tsv", sep="\t")
    jp = jp[jp.frame_source == "orf"].merge(catalog, on="junction_id", suffixes=("", "_c"))
    jp = jp[jp.frac_T >= 0.05]
    rows.append(jp.rename(columns={"class": "orf_class", "frac_T": "r3_frac_T", "cpm_T": "r3_cpm_T",
                                   "frac_Ntcga": "r3_frac_Ntcga", "frac_Ngtex": "r3_frac_Ngtex",
                                   "tumor_enriched": "r3_tumor_enriched"}).assign(
        junction_id=project + ":" + jp.junction_id, source=f"R3_{project}", jclass="cis_SAGe_" + jp.cls,
        r3_project=project))

published = pd.read_csv("results/cptac/benchmark/published_chimeric_peptides.tsv", sep="\t")
published = published[published.peptide_kind == "tryptic_MS"]
rows.append(pd.DataFrame({"peptide": published.peptide.str.upper().str.replace(r"[^A-Z]", "", regex=True),
                          "junction_id": "PUB", "source": "PUB", "jclass": published.chimera_type,
                          "name_d": published.gene_5p, "name_a": published.gene_3p, "pub_study": published.study}))
positives = pd.read_csv("results/cptac/poscontrol/positives.tsv", sep="\t")
rows.append(pd.DataFrame({"peptide": positives.peptide, "junction_id": "POS", "source": "POS", "jclass": "DNA_fusion",
                          "name_d": positives.fusion.str.split("-").str[0],
                          "name_a": positives.fusion.str.split("-").str[1]}))

groups = pd.concat(rows, ignore_index=True)[COLUMNS]
groups = groups[groups.peptide.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{7,40}", na=False)]
is_benchmark = groups.source.isin(["PUB", "POS"])
has_pseudogene = (groups.biotype_d.fillna("").str.contains("pseudogene")
                  | groups.biotype_a.fillna("").str.contains("pseudogene"))
in_family = groups.name_d.fillna("").str.contains(FAMILY_FILTER) | groups.name_a.fillna("").str.contains(FAMILY_FILTER)
groups["flag_family"] = (has_pseudogene | in_family) & ~is_benchmark
keep = sorted(set(groups.peptide) - (set(groups[groups.flag_family].peptide) - set(groups[is_benchmark].peptide)))

entrapments = shuffle_entrapments(keep)   # the sorted kept list fixes every entrapment

with open(f"{out}/db.fasta", "w") as fasta:
    fasta.write(open(REFERENCE).read())
    for i, p in enumerate(keep):
        fasta.write(f">CHIMPEP_{i}|{p}\n{p}\n")
    for i, (e, p) in enumerate(entrapments):
        fasta.write(f">ENTR_{i}|{p}\n{e}\n")
pd.Series(keep).to_csv(f"{out}/chim_peptides_kept.txt", index=False, header=False)
pd.DataFrame(entrapments, columns=["entrapment", "target"]).to_csv(f"{out}/entrapment.tsv", sep="\t", index=False)
groups.to_csv(f"{out}/peptide_groups.tsv", sep="\t", index=False)

# Flags. ref_IL: with I = L, the entry is in a reference protein; Sage keeps I/L variants apart at identical masses, so
# it wins a spectrum only by tie-breaking against its reference twin. bench_only: every source is PUB and/or POS (never
# a discovery). paralog_pair: a junction joins two genes sharing a >= 3-letter family prefix (annotation).
groups = groups[groups.peptide.isin(set(keep))]
junctions = groups[~groups.source.isin(["PUB", "POS"])]
root_d, root_a = (junctions[c].astype(str).str.extract(r"^([A-Z]{3,})", expand=False) for c in ("name_d", "name_a"))
flags = pd.DataFrame({"entry": keep})
flags["sources"] = flags.entry.map(groups.groupby("peptide").source.agg(lambda s: ",".join(sorted(set(s)))))
flags["ref_IL"] = flags.entry.isin(found_in(keep, [REFERENCE]))
flags["bench_only"] = flags.sources.fillna("").map(lambda s: set(s.split(",")) <= {"PUB", "POS"})
flags["paralog_pair"] = flags.entry.isin(junctions.peptide[root_d.notna() & (root_d == root_a)])
flags.to_csv(f"{out}/chimpep_flags.tsv", sep="\t", index=False)
