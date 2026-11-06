"""Option-B known-truth worlds: per fine fold (a family-closed chromosome group; coarse group = fold % 2) the reference
without the tryptic units of its held-out junctions plus PSEUDO, PENTR, NNEG, REJ and MOD; class caps; real-DB siblings.
usage: python -m juncta_t.worlds folds | siblings   -> optionb_v3/, caps_per_fold.json | siblings_real.tsv.gz"""
import collections
import itertools
import json
import os
import random
import re
import sys
from pathlib import Path

import ahocorasick
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from juncta.config import COHORTS, INPUTS
from juncta.peptide import REFERENCE, ends, found_in, read_fasta, read_gtf
from juncta_t.common import OB, RESULTS, SEARCH_MASS, TMT, il, sage_decoy

OPTIONB = f"{RESULTS}/optionb_v3"
E_DIRS = {"KIRC": "results/cptac/E_kirc", "UCEC": "results/cptac/pan/E/UCEC", "GBM": "results/cptac/pan/E/GBM"}
DB_V1 = "results/cptac/pan/db_v1"
R1_REF = "results/cptac/pan/cohorts/PDC000127/r1_ref.parquet"
# exact-isobaric residue-pair substitutions (monoisotopic masses equal to < 0.001 Da)
ISOBARIC = [("GG", "N"), ("AG", "Q"), ("GA", "Q"), ("AD", "EG"), ("DA", "GE"), ("AV", "GL"), ("VA", "LG")]
DEAMIDATION = 0.984016
FAMILY_RE = re.compile("^(" + "|".join(pd.read_csv(INPUTS / "gene_family_prefixes.tsv", sep="\t").prefix) + ")")
SIBLING_H2O, C13, SIBLING_TOL, SIBLING_WINDOW = 18.010565, 1.0033548, 20e-6, 4
MOD_OFFSETS = (0.0, DEAMIDATION, 2 * DEAMIDATION, TMT, TMT + DEAMIDATION, TMT + 2 * DEAMIDATION)
EXACT_SIBLING_RATE, EXACT_MOD_RATE = 0.019927, 0.013903   # among the disc CHIMPEP entries
SIBLING_COLUMNS = ["peptide", "sib_exact", "sib_near", "best_sib", "best_dppm", "best_win", "mod_any", "mod_exact",
                   "mod_det_ccrcc"]


def automaton_of(words):
    automaton = ahocorasick.Automaton()
    for word, value in words.items():
        automaton.add_word(word, value)
    automaton.make_automaton()
    return automaton


def assign_folds(jx, sequence_of_gene):
    """Genes joined by >= 4 shared 8-mers (I = L, each held by <= 4 genes: looser settings chain kinases through
    catalytic motifs) or by a name family stay on the group of their plurality chromosome."""
    genes_of_kmer = collections.defaultdict(set)
    for gene, sequence in sequence_of_gene.items():
        for i in range(len(sequence) - 7):
            word = il(sequence[i:i + 8])
            if len(set(word)) > 3:
                genes_of_kmer[word].add(gene)
    pairs = collections.Counter(pair for genes in genes_of_kmer.values() if 2 <= len(genes) <= 4
                                for pair in itertools.combinations(sorted(genes), 2))
    edges = [pair for pair, n in pairs.items() if n >= 4]
    for gene, name in jx.drop_duplicates("gene")[["gene", "name"]].itertuples(index=False):
        match = FAMILY_RE.match(str(name))
        if match:
            edges.append((gene, "FAM:" + re.sub(r"[-\d]+$", "", match.group(1))))
    parent = {}

    def root(x):
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    for a, b in edges:
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)
    jx = jx.assign(family=jx.gene.map(root))
    per_chromosome = jx.groupby(["family", "chr"]).size().rename("n").reset_index()
    per_chromosome = per_chromosome.sort_values(["n", "chr"], ascending=[False, True]).drop_duplicates("family")
    jx["fam_chr"] = jx.family.map(per_chromosome.set_index("family").chr)
    groups, totals = {}, np.zeros(OB["n_folds"])
    for chromosome, weight in jx.groupby("fam_chr").size().sort_values(ascending=False).items():
        groups[chromosome] = int(np.argmin(totals))   # greedy bin packing, largest first to the lightest group
        totals[groups[chromosome]] += weight
    jx["fold"] = jx.fam_chr.map(groups).astype(int)
    jx["coarse"] = jx.fold % 2
    return jx, pd.Series(groups, name="chrom_group").rename_axis("chr")


def gene_chromosomes():
    genes = read_gtf(["gene"], "gene_name")
    return pd.Series(genes.chr.values, index=genes.gene_name.values).groupby(level=0).first()


def cleavage_sites(sequence):
    sites = [i + 1 for i in range(len(sequence) - 1) if sequence[i] in "KR" and sequence[i + 1] != "P"]
    return [0] + sites + [len(sequence)]


def missed_cleavages(peptide):
    return len(cleavage_sites(peptide)) - 2


def excise_all(entries, peptides, min_piece=7):
    """Cuts the tryptic units around every occurrence (I = L) of {peptide: bond}, the junction after peptide[bond]."""
    automaton = automaton_of({il(p): (p, bond) for p, bond in peptides.items()})
    out = {}
    for header, sequence in entries.items():
        cuts = sorted({end - len(p) + 1 + b + 1 for end, (p, b) in automaton.iter(il(sequence))})
        if not cuts:
            out[header] = sequence
            continue
        sites = cleavage_sites(sequence)
        pieces, start = [], 0
        for cut in cuts:
            if start < cut < len(sequence):
                pieces.append(sequence[start:max(c for c in sites if c < cut)])
                start = min(c for c in sites if c > cut)
        kept = [x for x in pieces + [sequence[start:]] if len(x) >= min_piece]
        out.update({f"{header.split()[0]}#x{i}": piece for i, piece in enumerate(kept)})
    return out


def neighbor_profile(peptides, reference, skip_self, min_end=6, max_end=16):
    """Longest end of each peptide found in the reference; skip_self, ignoring whole-peptide hits, mimics excision."""
    peptides = [il(p) for p in peptides]
    words = collections.defaultdict(list)
    for i, q in enumerate(peptides):
        for k in range(min_end, min(len(q), max_end + 1)):
            words[q[:k]].append((i, 0, k))
            words[q[-k:]].append((i, 1, k))
    end_words = automaton_of({word: word for word in words})
    whole = automaton_of({q: i for i, q in enumerate(peptides)})
    longest = np.zeros((len(peptides), 2), int)
    for sequence in reference.values():
        q = il(sequence)
        own = {i for _, i in whole.iter(q)} if skip_self else set()
        for _, word in end_words.iter(q):
            for i, side, k in words[word]:
                if k > longest[i, side] and i not in own:
                    longest[i, side] = k
    return longest.max(1)


def placed(peptide, positions, residues):
    new = dict(zip(positions, residues))
    return "".join(new.get(i, a) for i, a in enumerate(peptide))


def kr_shuffle(peptide, rng):
    """The ENTR generator: K / R and the C-terminal residue stay, the other residues are shuffled."""
    free = [i for i, a in enumerate(peptide[:-1]) if a not in "KR"]
    residues = [peptide[i] for i in free]
    rng.shuffle(residues)
    return placed(peptide, free, residues)


def local_scramble(peptide, bond, rng, window=4):
    """Reverse or rotate a window of 3-4 residues next to the bond (K / R and the C-terminus fixed). Mass preserved."""
    length = len(peptide)
    lo = max(0, min(bond - window // 2 + 1, length - 1 - window))
    idx = [i for i in range(lo, min(length - 1, lo + window)) if peptide[i] not in "KR"]
    if len(idx) < 3:
        return None
    residues = [peptide[i] for i in idx]
    new = residues[::-1] if rng.random() < 0.5 else residues[1:] + residues[:1]
    if new == residues:
        new = residues[1:] + residues[:1] if new == residues[::-1] else residues[::-1]
    return None if new == residues else placed(peptide, idx, new)


def iso_swap(peptide, rng):
    """Replace one occurrence of one side of an exact-isobaric pair by the other. Mass preserved."""
    candidates = [(j, a, b) for a, b in ISOBARIC + [(y, x) for x, y in ISOBARIC]
                  for j in range(len(peptide) - len(a)) if peptide.startswith(a, j)]
    if not candidates:
        return None
    j, a, b = rng.choice(candidates)
    out = peptide[:j] + b + peptide[j + len(a):]
    return out if out != peptide and 7 <= len(out) <= 40 else None


def make_nneg(peptide_bonds, rng, forbid):
    """Scramble first for even i, substitution first for odd i, up to 8 tries each; the result must be I/L-new."""
    rows = []
    for i, (peptide, bond) in enumerate(peptide_bonds):
        generators = (("G1", lambda: local_scramble(peptide, bond, rng)), ("G2", lambda: iso_swap(peptide, rng)))
        for name, generate in (generators[::-1] if i % 2 else generators):
            new = next((x for x in (generate() for _ in range(8)) if x is not None and il(x) not in forbid), None)
            if new is not None:
                forbid.add(il(new))
                rows.append((new, peptide, name, "ref"))
                break
    return pd.DataFrame(rows, columns=["peptide", "source_peptide", "generator", "source_class"])


def make_mod(peptides, rng, forbid):
    """Even i: one N->D or Q->E, else as odd i: a terminal 2-4 residues become a 1-4 residue string of their mass + TMT
    over-label (S / T / Y / H) + 0-2 deamidations (5 ppm)."""
    segments = {}   # rounded search mass (0.001 Da) -> residue strings of length 1-4 (I excluded: I = L)
    for n in range(1, 5):
        for residues in itertools.product("GASPVTCLNDQKEMHFRYW", repeat=n):
            segments.setdefault(round(sum(SEARCH_MASS[x] for x in residues), 3), []).append("".join(residues))
    rows = []
    for i, peptide in enumerate(peptides):
        positions = [j for j, a in enumerate(peptide[:-1]) if a in "NQ"] if i % 2 == 0 else []
        if positions:
            j = rng.choice(positions)
            new = peptide[:j] + ("D" if peptide[j] == "N" else "E") + peptide[j + 1:]
            if il(new) not in forbid:
                forbid.add(il(new))
                rows.append((new, peptide, "M1_deam"))
                continue
        cterm = (i // 2) % 2 == 0
        for _ in range(6):
            k = rng.choice([2, 3, 4])
            if len(peptide) - k < 4:
                break
            old = peptide[-k:] if cterm else peptide[:k]
            if not any(x in "STYH" for x in peptide):
                break
            target = sum(SEARCH_MASS[x] for x in old) + TMT + DEAMIDATION * rng.choice([0, 1, 2])
            tolerance = 5e-6 * (sum(SEARCH_MASS[x] for x in peptide) + 18.010565 + TMT)
            candidates = [c for offset in (-0.002, -0.001, 0.0, 0.001, 0.002)
                          for c in segments.get(round(target + offset, 3), [])
                          if abs(sum(SEARCH_MASS[x] for x in c) - target) <= tolerance
                          and (not cterm or c[-1] in "KR") and c != old]
            if not candidates:
                continue
            segment = rng.choice(candidates)
            new = (peptide[:-k] + segment) if cterm else (segment + peptide[k:])
            if 7 <= len(new) <= 40 and il(new) not in forbid and missed_cleavages(new) <= 2:
                forbid.add(il(new))
                rows.append((new, peptide, "M2_overlabel_" + ("C" if cterm else "N")))
                break
    return pd.DataFrame(rows, columns=["peptide", "source_peptide", "generator"])


def half_half_join(pool, rng, forbid):
    """N-terminal part of one real tryptic peptide + C-terminal part (ending in K / R) of another."""
    rows = {}
    for _ in range(50 * OB["rej_eval"]):
        if len(rows) == OB["rej_eval"]:
            break
        a, b = rng.sample(pool, 2)
        if len(a) >= 7 and len(b) >= 5:
            joined = a[:rng.randint(3, len(a) - 3)] + b[rng.randint(1, len(b) - 3):]
            if (8 <= len(joined) <= 30 and joined[-1] in "KR" and il(joined) not in forbid and il(joined) not in rows
                    and missed_cleavages(joined) <= 2):
                rows[il(joined)] = (joined, a, b)
    return pd.DataFrame(rows.values(), columns=["peptide", "left_source", "right_source"])


def load_project(project):
    """In-frame junction-spanning control peptides of one project, with the bond of the junction in each peptide."""
    d = E_DIRS[project]
    core = pd.read_csv(f"{d}/junctions_core.tsv.gz", sep="\t", dtype={"chr_d": str})
    orf = pd.read_csv(f"{d}/orf/junction_orf.tsv.gz", sep="\t",
                      usecols=["junction_id", "orf_class", "orf_start", "junction_nt", "junction_aa"])
    proteins = pd.read_csv(f"{d}/orf/proteins.tsv.gz", sep="\t", usecols=["junction_id", "prot_chim"])
    peptides = pd.read_csv(f"{d}/orf/junction_peptides.tsv", sep="\t")
    peptides = peptides[(peptides.frame_source == "orf") & (peptides["class"] == "in_frame")
                        & (peptides.novelty == "junction_spanning") & peptides.in_reference]
    unique = set(Path(f"{d}/orf/control_peptides_unique.txt").read_text().split())
    j = core.merge(orf, on="junction_id").merge(proteins, on="junction_id")
    j = j[(j.orf_class == "in_frame") & j.prot_chim.notna()]
    rows = []
    columns = ["junction_id", "prot_chim", "junction_aa", "orf_start", "junction_nt"]
    for r in peptides.merge(j[columns], on="junction_id").itertuples(index=False):
        junction_aa = int(r.junction_aa)
        offset = r.prot_chim.find(r.peptide)
        while offset >= 0 and not offset <= junction_aa - 1 <= offset + len(r.peptide) - 2:
            offset = r.prot_chim.find(r.peptide, offset + 1)
        if offset >= 0:
            rows.append((r.junction_id, r.peptide, junction_aa - 1 - offset, offset, r.peptide in unique,
                         int((r.junction_nt - r.orf_start) % 3)))
    table = pd.DataFrame(rows, columns=["junction_id", "peptide", "bond", "offset", "unique", "codon_phase"])
    j["junction_key"] = [f"{c}:{d + 2 if s == '+' else a + 2}-{a if s == '+' else d}:{s}"
                         for c, d, a, s in zip(j.chr_d, j.pos_d, j.pos_a, j.str_d)]
    table = table.merge(j[["junction_id", "junction_key", "chr_d", "gene_d", "name_d", "prot_chim"]], on="junction_id")
    return table.assign(project=project)


def accession_gene_map():
    genes = {}
    for name in ("UP000005640_9606", "UP000005640_9606_additional"):
        for header, _ in read_fasta(f"data/uniprot_ref/{name}.fasta.gz"):
            match = re.search(r"GN=(\S+)", header)
            if match:
                genes[header.split("|")[1]] = match.group(1)
    return genes


def gene_of_accession(accession, genes):
    """GC|ENSP|GENE and ENS|ENSP|GENE headers of ref_pq.fasta carry the gene; sp and tr go through the UniProt map."""
    fields = accession.removeprefix("rev_").split("#x")[0].split("|")
    if len(fields) >= 3 and fields[0] in ("GC", "ENS"):
        return fields[2]
    if len(fields) >= 2:
        return genes.get(fields[1].split("-")[0]) or genes.get(fields[1])
    return None


def matching_weights(jx, pep, reference, genes):
    """Tempered density ratios chimera / control over five peptide features, junction mean, times a low-abundance boost.
    RNA abundance is matched on the chimeric candidates with tissue RNA only."""
    groups = pd.read_csv(f"{DB_V1}/peptide_groups.tsv", sep="\t",
                         usecols=["peptide", "junction_id", "source", "name_d", "name_a"])
    chim = groups[~groups.source.isin(["PUB", "POS"])].drop_duplicates("peptide")
    cpm = pd.read_csv("results/cptac/global_tier/chim_junction_kirc_cpm.tsv", sep="\t",
                      usecols=["junction_id", "cpm_T"])
    chim = chim.merge(cpm.drop_duplicates("junction_id"), on="junction_id", how="left")
    chim = chim.iloc[np.random.default_rng(OB["seed"]).permutation(len(chim))[:12000]]
    ctrl = pep.drop_duplicates("peptide")
    psms = pq.read_table(R1_REF, columns=["proteins", "spectrum_q"], filters=[("spectrum_q", "<=", 0.01)])
    counts = psms.column("proteins").to_pandas().value_counts()
    abundance = counts.groupby([gene_of_accession(a.split(";")[0], genes) for a in counts.index]).sum()
    ctrl = pd.DataFrame({"peptide": ctrl.peptide.values, "junction_key": ctrl.junction_key.values,
                         "cpm": ctrl.junction_key.map(jx.set_index("junction_key").cpm_KIRC).values,
                         "abund": ctrl.name_d.map(abundance).fillna(0).values,
                         "shared_end": neighbor_profile(ctrl.peptide, reference, skip_self=True)})
    chim = pd.DataFrame({"peptide": chim.peptide.values, "cpm": chim.cpm_T.fillna(0).values,
                         "abund": np.maximum(chim.name_d.map(abundance).fillna(0).values,
                                             chim.name_a.map(abundance).fillna(0).values),
                         "shared_end": neighbor_profile(chim.peptide, reference, skip_self=False)})
    for table in (ctrl, chim):
        table["length"] = table.peptide.str.len()
        table["mc"] = table.peptide.map(missed_cleavages)
    bins = {"cpm": lambda x: np.digitize(np.log10(x + 1e-4), [-2.5, -1.5, -0.5, 0.5, 1.5]),
            "length": lambda x: np.digitize(x, [10, 14, 18, 24]),
            "mc": lambda x: np.minimum(x, 2),
            "shared_end": lambda x: np.digitize(x, [6, 8, 11, 15]),
            "abund": lambda x: np.digitize(np.log10(x + 1), [0.5, 1.5, 2.5, 3.5])}
    log_weight = np.zeros(len(ctrl))
    for feature, binning in bins.items():
        bin_ctrl = binning(ctrl[feature].values)
        bin_chim = binning((chim[chim.cpm > 0] if feature == "cpm" else chim)[feature].values)
        k = max(bin_ctrl.max(), bin_chim.max()) + 1
        p_ctrl = np.bincount(bin_ctrl, minlength=k) / len(bin_ctrl)
        p_chim = np.bincount(bin_chim, minlength=k) / len(bin_chim)
        log_weight += OB["match_tau"] * np.log(((p_chim + 0.01) / (p_ctrl + 0.01))[bin_ctrl])
    ctrl["w"] = np.exp(log_weight) * np.where(bins["abund"](ctrl.abund.values) <= 1, OB["low_abund_boost"], 1.0)
    return ctrl.groupby("junction_key").w.mean()


def absent(table):
    return table[~table.peptide.isin(found_in(list(table.peptide), [REFERENCE]))].reset_index(drop=True)


def build_fold(fold, jx, pep, weight, reference, all_detected, chim_entr):
    out = f"{OPTIONB}/fold{fold}"
    os.makedirs(out, exist_ok=True)
    # weighted sampling without replacement (Efraimidis-Spirakis keys) inside the fold's group
    candidates = jx[jx.fold == fold]
    w = candidates.junction_key.map(weight).fillna(weight.median()).values
    key = np.log(np.random.default_rng(OB["seed"] + fold).random(len(candidates))) / w
    selected = candidates.junction_key.iloc[np.argsort(-key)[:int(round(OB["fold_frac"] * len(jx)))]]
    held = pep[pep.junction_key.isin(selected)].drop_duplicates("peptide").reset_index(drop=True)
    reduced = excise_all(reference, dict(zip(held.peptide, held.bond)))
    held_il = {il(p) for p in held.peptide}
    forbid = chim_entr | held_il   # the generated sequences stay outside CHIMPEP, ENTR and PSEUDO
    rng = random.Random(OB["seed"] + 100 * fold)
    pentr = []
    for i, peptide in enumerate(held.peptide):
        shuffled = next((x for x in (kr_shuffle(peptide, rng) for _ in range(200)) if il(x) not in forbid), None)
        if shuffled is not None:
            forbid.add(il(shuffled))
            pentr.append((i, shuffled, peptide))
    pentr = pd.DataFrame(pentr, columns=["pseudo_idx", "peptide", "target"])
    detected = all_detected[all_detected.grp == fold]
    detected = detected[pd.qcut(detected.n.rank(method="first"), 10, labels=False) <= 8]   # no top abundance decile
    pick = detected.sample(n=min(OB["nneg_ref"], len(detected)), random_state=OB["seed"] + fold)
    nneg = make_nneg(((p, len(p) // 2) for p in pick.index), rng, forbid | held_il)   # a copy: REJ and MOD ignore NNEG
    rej = half_half_join(list(detected.index[:20000]), rng, forbid)
    rest = detected[~detected.index.isin(pick.index)]
    n_sources = min(int(OB["n_mod"] * 1.3), len(rest))
    sources = rest.sample(n=n_sources, random_state=OB["seed"] + 1000 + fold).index.tolist()
    mod = make_mod(sources, rng, forbid)
    pentr, nneg, rej, mod = absent(pentr), absent(nneg), absent(rej), absent(mod).head(OB["n_mod"])
    held["pseudo_id"] = [f"PSEUDO_{i}" for i in range(len(held))]
    records = list(reduced.items())
    records += [(f"{r.pseudo_id}|{r.name_d}|{r.junction_key}", r.peptide) for r in held.itertuples()]
    records += [(f"PENTR_{r.pseudo_idx}|{r.target}", r.peptide) for r in pentr.itertuples()]
    records += [(f"NNEG_{i}|{r.generator}|{r.source_class}|{r.source_peptide}", r.peptide)
                for i, r in enumerate(nneg.itertuples())]
    records += [(f"REJ_{i}", r.peptide) for i, r in enumerate(rej.itertuples())]
    records += [(f"MOD_{i}|{r.generator}|{r.source_peptide}", r.peptide) for i, r in enumerate(mod.itertuples())]
    Path(f"{out}/db.fasta").write_text("".join(f">{header}\n{sequence}\n" for header, sequence in records))
    held.drop(columns=["prot_chim"]).to_csv(f"{out}/pseudo.tsv", sep="\t", index=False)
    for name, table in (("pentr", pentr), ("nneg", nneg), ("rej", rej), ("mod", mod)):
        table.to_csv(f"{out}/{name}.tsv", sep="\t", index=False)
    # longest prefix and suffix (>= 3 aa, I = L) of every novel entry and its Sage decoy found in a reference entry
    novel = set(held.peptide).union(pentr.peptide, nneg.peptide, rej.peptide, mod.peptide)
    entries = sorted(novel | {sage_decoy(p) for p in novel})
    words = {w for q in map(il, entries) for k in range(3, len(q) + 1) for w in (q[:k], q[-k:])}
    automaton = automaton_of({word: word for word in words})
    seen = {word for sequence in reduced.values() for _, word in automaton.iter(il(sequence))}
    shared = pd.DataFrame([(p, *ends(p, seen)) for p in entries], columns=["entry", "L_pre", "L_suf"])
    shared.to_csv(f"{out}/shared_ends.tsv", sep="\t", index=False)


def sibling_mass(peptide):
    return sum(SEARCH_MASS[a] for a in peptide) + SIBLING_H2O + TMT


def ref_index(fasta, is_ref):
    """I/L-normalised tryptic peptides (<= 2 missed cleavages, 7-40 aa) of the reference entries, sorted by mass."""
    peptides = set()
    for header, sequence in read_fasta(fasta):
        if is_ref(header):
            bounds = cleavage_sites(sequence)
            for i in range(len(bounds) - 1):
                peptides.update(sequence[bounds[i]:bounds[j]] for j in range(i + 1, min(i + 4, len(bounds)))
                                if 7 <= bounds[j] - bounds[i] <= 40)
    # a fixed order makes the tie-break of best_sib reproducible
    keys = np.array(list(dict.fromkeys(il(p) for p in sorted(peptides) if set(p) <= set(SEARCH_MASS))), dtype=object)
    masses = np.array([sibling_mass(p) for p in keys])
    order = np.argsort(masses)
    return keys[order], masses[order]


def siblings(query, keys, masses, detected_ccrcc=(), detected_pan=()):
    """A sibling is a reference peptide within 20 ppm at isotope offset -1, 0 or 1 differing in one window of <= 4
    residues; a mimic the same with M(query) shifted by deamidation or TMT over-label."""
    normalised = il(query)
    M = sibling_mass(query)
    best = None
    sib_exact = sib_near = mod_any = mod_exact = mod_detected = 0
    for offset in MOD_OFFSETS:
        for k in (-1, 0, 1):
            target = M - offset - k * C13
            lo = np.searchsorted(masses, target * (1 - SIBLING_TOL))
            hi = np.searchsorted(masses, target * (1 + SIBLING_TOL), side="right")
            for r, reference_mass in zip(keys[lo:hi], masses[lo:hi]):
                prefix = len(os.path.commonprefix([normalised, r]))
                suffix = len(os.path.commonprefix([normalised[::-1], r[::-1]]))
                suffix = min(suffix, min(len(normalised), len(r)) - prefix)
                window = max(len(normalised), len(r)) - prefix - suffix
                if r == normalised or window > SIBLING_WINDOW:
                    continue
                exact = k == 0 and abs(M - offset - reference_mass) < 0.002
                if offset > 0:
                    mod_any = 1
                    mod_exact |= exact
                    mod_detected |= r in detected_ccrcc
                    continue
                sib_exact |= exact
                sib_near |= not exact
                score = (r in detected_pan, r in detected_ccrcc, exact, -window)
                if best is None or score > best[0]:
                    best = (score, r, (M - k * C13 - reference_mass) / reference_mass * 1e6, window)
    return (query, int(sib_exact), int(sib_near), *(best[1:] if best else ("", np.nan, -1)), mod_any, int(mod_exact),
            int(mod_detected))


def measure_real():
    """Sibling columns of every CHIMPEP and ENTR entry; 'detected' = >= 3 rank-1 PSMs at q <= 0.001."""
    detected = {}
    for cohort in COHORTS:
        table = pq.read_table(f"results/cptac/pan/cohorts/{cohort}/r1_ref.parquet",
                              columns=["stripped_peptide", "spectrum_q"], filters=[("spectrum_q", "<=", 0.001)])
        counts = table.to_pandas().stripped_peptide.str.replace("I", "L").value_counts()
        detected[cohort] = set(counts.index[counts >= 3])
    keys, masses = ref_index(f"{DB_V1}/db.fasta", lambda h: not h.startswith(("CHIMPEP_", "ENTR_")))
    queries = set(pd.read_csv(f"{DB_V1}/chimpep_flags.tsv", sep="\t").entry)
    queries |= set(pd.read_csv(f"{DB_V1}/entrapment.tsv", sep="\t").entrapment)
    pan = set().union(*detected.values())
    rows = [siblings(q, keys, masses, detected["PDC000127"], pan) for q in sorted(queries)]
    out = f"{RESULTS}/siblings_mod"
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows, columns=SIBLING_COLUMNS).to_csv(f"{out}/siblings_real.tsv.gz", sep="\t", index=False)


def fold_caps():
    """NNEG and MOD entries the class needs for the exact-sibling and exact-mimic rates of CHIMPEP; MOD is reference."""
    caps = {}
    for fold in range(OB["n_folds"]):
        fasta = f"{OPTIONB}/fold{fold}/db.fasta"
        keys, masses = ref_index(fasta, lambda h: not h.startswith(("PSEUDO_", "PENTR_", "NNEG_", "REJ_")))
        rows = [siblings(q, keys, masses) for h, q in read_fasta(fasta) if h.startswith(("PSEUDO_", "REJ_"))]
        table = pd.DataFrame(rows, columns=SIBLING_COLUMNS)
        n_cap = (EXACT_SIBLING_RATE * len(table) - table.sib_exact.sum()) / (1 - EXACT_SIBLING_RATE)
        n_cap_mod = (EXACT_MOD_RATE * len(table) - table.mod_exact.sum()) / (1 - EXACT_MOD_RATE)
        caps[fold] = dict(n_cap=max(0, round(n_cap)), n_cap_mod=max(0, round(n_cap_mod)))
    os.makedirs(f"{RESULTS}/siblings", exist_ok=True)
    with open(f"{RESULTS}/siblings/caps_per_fold.json", "w") as fh:
        json.dump(caps, fh, indent=1)


def build_worlds():
    os.makedirs(f"{OPTIONB}/pool", exist_ok=True)
    pool = pd.concat([load_project(p) for p in E_DIRS], ignore_index=True)
    # one row per (junction, peptide); the bond of a peptide within a junction is identical across projects
    pool = pool.sort_values(["junction_key", "peptide", "project"]).drop_duplicates(["junction_key", "peptide"])
    jx = pool.groupby("junction_key").agg(gene=("gene_d", "first"), name=("name_d", "first"), chr=("chr_d", "first"),
                                          n_pep=("peptide", "size"), n_unique=("unique", "sum"),
                                          projects=("project", lambda x: ",".join(sorted(set(x))))).reset_index()
    jx = jx[jx.n_unique > 0].copy()   # eligible: at least one junction-specific spanning peptide
    pool = pool[pool.junction_key.isin(jx.junction_key)]
    for tcga, directory in E_DIRS.items():
        table = pd.read_csv(f"{directory}/canonical_all.tsv.gz", sep="\t", dtype={"chr": str})
        keys = table.chr + ":" + table.start.astype(str) + "-" + table.end.astype(str) + ":" + table.strand
        jx[f"cpm_{tcga}"] = jx.junction_key.map(table.set_index(keys).cpm_T).fillna(0.0).values
    jx, chromosome_group = assign_folds(jx, pool.drop_duplicates("gene_d").set_index("gene_d").prot_chim)
    chromosome_group.to_csv(f"{OPTIONB}/pool/chrom_groups.tsv", sep="\t")
    reference = dict(read_fasta(REFERENCE))
    genes = accession_gene_map()
    weight = matching_weights(jx, pool, reference, genes)
    jx = jx.assign(w=jx.junction_key.map(weight))
    jx.to_csv(f"{OPTIONB}/pool/junctions.tsv.gz", sep="\t", index=False)
    # reference peptides of <= 30 aa with >= 3 CCRCC PSMs at q <= 0.001 and the chromosome group of their gene
    r1 = pq.read_table(R1_REF, columns=["stripped_peptide", "proteins", "spectrum_q"],
                       filters=[("spectrum_q", "<=", 0.001)]).to_pandas()
    detected = r1.groupby("stripped_peptide").agg(n=("proteins", "size"), acc=("proteins", "first"))
    detected = detected[(detected.n >= 3) & (detected.index.str.len() <= 30)]
    detected["gene"] = [gene_of_accession(a.split(";")[0], genes) for a in detected.acc]
    detected["grp"] = detected.gene.map(gene_chromosomes()).map(chromosome_group)
    chim_entr = {il(s) for h, s in read_fasta(f"{DB_V1}/db.fasta") if h.startswith(("CHIMPEP_", "ENTR_"))}
    for fold in range(OB["n_folds"]):
        build_fold(fold, jx, pool, weight, reference, detected, chim_entr)
    fold_caps()


if __name__ == "__main__":
    measure_real() if sys.argv[1] == "siblings" else build_worlds()
