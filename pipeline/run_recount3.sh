#!/bin/bash
# RNA sources of one TCGA project: recount3 download, cis-SAGe catalog, ORFs, canonical-junction controls and peptides.
# usage: run_recount3.sh <TCGA project> <GTEx tissue> [out_dir] [control_dir]: KIRC KIDNEY results/cptac/recount3_kirc
# results/cptac/E_kirc, BRCA BREAST, COAD COLON, GBM BRAIN, HNSC SALIVARY_GLAND, LUAD LUNG, LUSC LUNG, OV OVARY, PAAD
# PANCREAS, UCEC UTERUS.
set -euo pipefail
PROJECT=$1; TISSUE=$2; OUT=${3:-results/cptac/pan/r3/$PROJECT}; CONTROLS=${4:-results/cptac/pan/E/$PROJECT}
HERE=$(dirname "$0"); BASE=https://recount-opendata.s3.amazonaws.com/recount3/release/human/data_sources
TCGA=data/recount3/tcga_$PROJECT; GTEX=data/recount3/gtex_$TISSUE
mkdir -p "$TCGA" "$GTEX"
p=${PROJECT: -2}; t=${TISSUE: -2}
for k in RR ID MM; do
  curl -sSfL --output-dir "$TCGA" -O "$BASE/tcga/junctions/$p/$PROJECT/tcga.junctions.$PROJECT.UNIQUE.$k.gz"
  curl -sSfL --output-dir "$GTEX" -O "$BASE/gtex/junctions/$t/$TISSUE/gtex.junctions.$TISSUE.UNIQUE.$k.gz"
done
curl -sSfL --output-dir "$TCGA" -O "$BASE/tcga/metadata/$p/$PROJECT/tcga.tcga.$PROJECT.MD.gz"
python "$HERE/02_junctions.py" recount3 "$OUT" "$TCGA" "$GTEX"
python "$HERE/03_orf.py" "$OUT/junctions_core.tsv.gz" "$OUT/orf"
python "$HERE/02_junctions.py" controls "$CONTROLS" "$TCGA" "$OUT/cis_sage_catalog.tsv.gz"
python "$HERE/03_orf.py" "$CONTROLS/junctions_core.tsv.gz" "$CONTROLS/orf" keep
