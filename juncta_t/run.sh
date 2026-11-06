#!/bin/bash
# Juncta-T final run, from the working directory holding data/ and results/ with the outputs of pipeline/.
# MAIN_PY: envs/juncta.yml with sage on PATH, ~100 GB RAM. DL_PY: envs/juncta_dl.yml. GPU_PY: one GPU, torch 2.7.1, numpy, pyyaml.
# Needs the MGFs of the three option-B cohorts.
set -euo pipefail
MAIN=${MAIN_PY:-python}
DL=${DL_PY:-python}
GPU=${GPU_PY:-python}
R=results/cptac/dl
OPTIONB="PDC000127 PDC000125 PDC000204"
FOLDS="0 1 2 3 4 5"
ROUNDS="R3_ET0 R1b_ETk"

$MAIN -m juncta_t.worlds folds
$MAIN -m juncta_t.worlds siblings
for P in $OPTIONB; do
  for F in $FOLDS; do
    S=$R/optionb_v3/sage/${P}_f$F
    $MAIN pipeline/06_search.py data/cptac_mgf/$P $R/optionb_v3/fold$F/db.fasta $S $S/reporter.parquet
    $MAIN -m juncta_t.shards fold $P $F
  done
  $MAIN -m juncta_t.shards pretrain $P
done
for P in $OPTIONB PDC000153 PDC000116 PDC000234 PDC000221 PDC000270 PDC000120 PDC000110; do
  $MAIN -m juncta_t.shards production $P
done

for G in 0 1; do for H in 0 1; do
  $GPU -m juncta_t.model_spectrum train $G $H
  $GPU -m juncta_t.model_spectrum score $G $H
done; done
for P in $OPTIONB; do for F in $FOLDS; do
  $MAIN -m juncta_t.shards stitch $P $F
  OMP_NUM_THREADS=2 $DL -m juncta_t.rescore_b2 $P $F   # DeepLC output depends on the thread counts
done; done

for F in $FOLDS; do for H in 0 1; do
  $DL -m juncta_t.model_entry unit full $F $H
  $DL -m juncta_t.model_entry unit full_trunc $F $H
done; done
for H in 0 1; do $DL -m juncta_t.model_entry unit prod $H; done
for ROUND in $ROUNDS; do
  for G in 0 1; do for H in 0 1; do for SEED in 0 1 2; do
    $DL -m juncta_t.model_entry train $ROUND $G $H $SEED
  done; done; done
  $DL -m juncta_t.model_entry score $ROUND
done

for SET in dev_f0123 test_f45 dev_f0123_trunc test_f45_trunc; do
  $DL -m juncta_t.evaluate psm $SET
  for ROUND in $ROUNDS; do $DL -m juncta_t.evaluate $ROUND $SET; done
done
for ROUND in $ROUNDS; do $DL -m juncta_t.evaluate prod $ROUND; done
$DL -m juncta_t.evaluate e2

