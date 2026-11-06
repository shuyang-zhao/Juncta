"""FASTA and GTF reading, tryptic sites, entrapments, residue masses and the neighbour-ion rule: a peptide passes with
>= 2 distinct cleavage sites matched beyond its longest reference-shared prefix and suffix (>= 3 aa, I = L). Ions carry
TMT on the N-terminus and K, Cam C and the Met oxidations of the PSM, and match as 1+ (and 2+ when z >= 3) among top
peaks."""
import gzip
import random
import re

import numpy as np
import pandas as pd

REFERENCE = "data/cptac_ref/ref_pq.fasta"

# Monoisotopic residue masses of the rule; C includes carbamidomethyl.
AA = dict(G=57.021464, A=71.037114, S=87.032028, P=97.052764, V=99.068414, T=101.047679, C=160.030649, L=113.084064,
          I=113.084064, N=114.042927, D=115.026943, Q=128.058578, K=128.094963, E=129.042593, M=131.040485,
          H=137.058912, F=147.068414, R=156.101111, Y=163.063329, W=186.079313)
LAB, PROTON, H2O, OX = 229.162932, 1.007276, 18.010565, 15.994915


def read_fasta(path):
    """Yield (header, sequence) of a plain or gzipped FASTA."""
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as fh:
        for record in fh.read()[1:].split("\n>"):
            header, _, seq = record.partition("\n")
            yield header.strip(), seq.replace("\n", "")


def read_gtf(features, *attributes):
    """Rows of the Ensembl 91 GTF with these features, with the named attributes as columns."""
    gtf = pd.read_csv("data/ref/Homo_sapiens.GRCh38.91.gtf", sep="\t", comment="#", header=None, dtype={"chr": str},
                      usecols=[0, 2, 3, 4, 6, 8], names=["chr", "feat", "start", "end", "strand", "attr"])
    gtf = gtf[gtf.feat.isin(features)].copy()
    for name in attributes:
        gtf[name] = gtf.attr.str.extract(name + r' "([^"]+)"', expand=False)
    return gtf


def matches(peptides, paths):
    """Yield (header, peptide) of every occurrence (I = L), for each input peptide of an I/L-equivalent group."""
    import ahocorasick   # absent from the deep-learning environment, which only needs the masses

    groups = {}
    for p in peptides:
        groups.setdefault(p.replace("I", "L"), set()).add(p)
    if not groups:
        return
    automaton = ahocorasick.Automaton()
    for key, group in groups.items():
        automaton.add_word(key, sorted(group))
    automaton.make_automaton()
    for path in paths:
        for header, seq in read_fasta(path):
            for _, group in automaton.iter(seq.replace("I", "L")):
                for p in group:
                    yield header, p


def found_in(peptides, paths):
    return {p for _, p in matches(peptides, paths)}


def reference_pieces(peptides, fasta=REFERENCE):
    """The prefixes and suffixes (>= 3 aa, I = L) of the peptides found in a reference protein; the input of ends()."""
    words = set()
    for p in peptides:
        q = p.replace("I", "L")
        for k in range(3, len(q) + 1):
            words.update((q[:k], q[-k:]))
    return found_in(words, [fasta])


def ends(p, pieces):
    """Lengths of the longest reference-shared prefix and suffix of p."""
    q = p.replace("I", "L")
    return (max([k for k in range(3, len(q) + 1) if q[:k] in pieces], default=0),
            max([k for k in range(3, len(q) + 1) if q[-k:] in pieces], default=0))


def cleavage_sites(p):
    """Offsets of the tryptic cleavages inside p: after K or R, except before P."""
    return [i + 1 for i in range(len(p) - 1) if p[i] in "KR" and p[i + 1] != "P"]


def shuffle_entrapments(peptides, seed=20261103):
    """(entrapment, peptide) pairs: a shuffle that keeps K, R, each P after them and the C-terminus, so the entrapment
    digests like its peptide, and that is no peptide or earlier entrapment (I = L). One random stream walks the list."""
    targets = {p.replace("I", "L") for p in peptides}
    rng = random.Random(seed)
    used, pairs = set(), []
    for p in peptides:
        free = [i for i in range(len(p) - 1) if p[i] not in "KR" and not (p[i] == "P" and i and p[i - 1] in "KR")]
        for _ in range(200):
            residues = [p[i] for i in free]
            rng.shuffle(residues)
            shuffled = list(p)
            for i, a in zip(free, residues):
                shuffled[i] = a
            e = "".join(shuffled)
            key = e.replace("I", "L")
            if key not in targets and key not in used and cleavage_sites(e) == cleavage_sites(p):
                used.add(key)
                pairs.append((e, p))
                break
    return pairs


def parse_modified(modified):
    """Sequence and 0-based oxidised-Met positions of a Sage peptide, as [+229.16293]-AM[+15.994915]K[+229.16293]."""
    residues = re.findall(r"([A-Z])(\[[^]]*\])?", modified)
    oxidised = [i for i, (a, mod) in enumerate(residues) if a == "M" and mod.startswith("[+15.99")]
    return "".join(a for a, _ in residues), oxidised


def ions(p, oxidised=()):
    """b ions, y ions (1+) and the neutral mass of the TMT-labelled peptide with these 0-based oxidised positions."""
    m = np.array([AA[a] + (LAB if a == "K" else 0) for a in p])
    m[list(oxidised)] += OX
    return np.cumsum(m)[:-1] + LAB + PROTON, np.cumsum(m[::-1])[:-1] + H2O + PROTON, m.sum() + LAB + H2O


def matched_ions(modified, z, mz, tol=0.02):
    """Sets of matched b and y ion numbers of a Sage peptide in the peak list mz; the default tolerance is Orbitrap MS2,
    in Da."""
    b, y, _ = ions(*parse_modified(modified))
    hit_b, hit_y = set(), set()
    for hits, series in ((hit_b, b), (hit_y, y)):
        for i, v in enumerate(series, start=1):
            for c in [v] + ([(v + PROTON) / 2] if z >= 3 else []):
                if (np.abs(mz - c) <= tol).any():
                    hits.add(i)
    return hit_b, hit_y


def specific_sites(p, hit_b, hit_y, prefix, suffix):
    """Distinct cleavage sites matched beyond the reference-shared ends."""
    return {i for i in hit_b if i > prefix} | {len(p) - j for j in hit_y if j > suffix}
