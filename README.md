# Juncta

Juncta finds protein products of chimeric RNAs in large TMT proteomics cohorts.

It searches spectra against a database of chimeric junction peptides with matched entrapment peptides, controls the false discovery rate within the chimeric class using a decoy-based null with a generalized Pareto tail, and combines cohorts in one pan-cancer test. Candidates are then checked with PepQuery and a fragment-ion rule at the junction.

**Juncta-T** is a transformer that rescores peptide candidates: a peak-level spectrum encoder feeding an entry-level set transformer. It is trained and evaluated on known-truth worlds, in which reference junctions are hidden from the database and re-inserted as novel entries. Evaluation is cross-fitted: each entry is scored by models that saw neither its genes nor its plex half. Folds 4 and 5 were kept out of model selection, but they are not a sealed hold-out, because the other folds' models train on them.

## Repository layout

| Path | Contents |
|---|---|
| `juncta/` | shared library: cohorts, null model, entry matching, peptide and spectrum helpers |
| `pipeline/` | the screen, as numbered steps from reference proteome to main table |
| `juncta_t/` | Juncta-T: known-truth worlds, models, evaluation |
| `inputs/` | small curated tables |
| `envs/` | conda environments |

## Installation

```bash
git clone https://github.com/shuyang-zhao/Juncta.git && cd Juncta
conda env create -f envs/juncta.yml       # screen
conda env create -f envs/juncta_dl.yml    # Juncta-T
pip install -e .                          # in each environment
```

[PepQuery](http://www.pepquery.org) is downloaded from its authors; set `PEPQUERY_JAR` to its jar file.

## Usage

Run from the repository root, with input data in `data/` and `results/`. Cohorts are defined in `juncta/config.py`.

```bash
# Database
python pipeline/01_reference.py
python pipeline/02_junctions.py lalani results/cptac/lalani
python pipeline/03_orf.py results/cptac/lalani/junctions_core.tsv.gz results/cptac/lalani/orf
pipeline/run_recount3.sh KIRC KIDNEY results/cptac/recount3_kirc results/cptac/E_kirc   # projects are listed in the script
python pipeline/04_union_db.py results/cptac/pan/db_v1

# Search and statistics
pipeline/run_cohort.sh PDC000127     # once per cohort
python pipeline/09_pan.py
python pipeline/10_pepquery.py results/cptac/pan/candidate_rows.tsv results/cptac/pan/pepquery_hits --entrapment
python pipeline/11_main_table.py     # writes results/cptac/pan/main_hits.tsv

# Juncta-T (the script header lists the environments)
bash juncta_t/run.sh
```

Each script's docstring gives its usage. Run the tests with `pytest tests/`; `tests/test_pipeline.py` builds a small synthetic screen and runs steps 08 to 11 on it, which also shows the files those steps read.

## Data

Spectra come from the [Proteomic Data Commons](https://pdc.cancer.gov), junctions from [recount3](https://rna.recount.bio) and published chimeric RNA catalogues, and reference sequences from UniProt, Ensembl and GENCODE.

The scripts download spectra and recount3 junctions themselves. Reference files go in `data/` under the paths named in `pipeline/01_reference.py` and `pipeline/03_orf.py`. These project tables are inputs that no script here produces:

| File | Columns read | Content | Read by |
|---|---|---|---|
| `results/cptac/pepquery/input_drs_tierAB/peptide_map.tsv` | `peptide junction_id name_d name_a orf_class category novelty` | junction peptides of a direct-RNA-sequencing chimera catalogue | `04_union_db.py` |
| `results/master/chrna_master.tsv.gz` | `junction_id biotype_d biotype_a category` | gene biotypes of those junctions | `04_union_db.py` |
| `results/cptac/benchmark/published_chimeric_peptides.tsv` | `peptide peptide_kind chimera_type gene_5p gene_3p study` | chimeric peptides curated from the literature; `peptide_kind` `tryptic_MS` is used | `04_union_db.py` |
| `results/cptac/poscontrol/positives.tsv` | `peptide fusion` | peptides of DNA-fusion positive controls | `04_union_db.py` |
| `results/cptac/global_tier/chim_junction_kirc_cpm.tsv` | `junction_id cpm_T` | tumour RNA abundance of the chimeric junctions | `juncta_t/worlds.py` |

## Citation

Paper in preparation.

## License

[MIT](LICENSE)
