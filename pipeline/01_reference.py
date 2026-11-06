"""Reference proteome (UniProt with TrEMBL, Ensembl 91, GENCODE v50, cRAP; the first source wins on equal sequences),
cleaned for PepQuery's FM index: sequences split at every non-standard residue or '*', pieces < 7 aa dropped."""
import os
import re

from juncta.peptide import REFERENCE, read_fasta

first_name = {}
for path in ["data/uniprot_ref/UP000005640_9606.fasta.gz", "data/uniprot_ref/UP000005640_9606_additional.fasta.gz"]:
    for header, seq in read_fasta(path):
        first_name.setdefault(seq, header.split()[0])
for header, seq in read_fasta("data/ensembl91_pep/Homo_sapiens.GRCh38.pep.all.fa.gz"):
    symbol = re.search(r"gene_symbol:(\S+)", header)
    first_name.setdefault(seq.rstrip("*"), f"ENS|{header.split()[0]}|{symbol.group(1) if symbol else 'NA'}")
for header, seq in read_fasta("data/dk/gencode.v50.pc_translations.fa.gz"):
    fields = header.split("|")
    first_name.setdefault(seq, f"GC|{fields[0]}|{fields[6]}")
reference = [(name, seq) for seq, name in first_name.items()]
reference += [(f"CRAP_{header.split()[0]}", seq) for header, seq in read_fasta("data/crap/crap.fasta")]

os.makedirs(os.path.dirname(REFERENCE), exist_ok=True)
with open(REFERENCE, "w") as cleaned:
    for name, seq in reference:
        pieces = [p for p in re.split(r"[^ACDEFGHIKLMNPQRSTVWY]+", seq.upper()) if len(p) >= 7]
        for i, piece in enumerate(pieces):
            cleaned.write(f">{name}{f'_part{i + 1}' if len(pieces) > 1 else ''}\n{piece}\n")
