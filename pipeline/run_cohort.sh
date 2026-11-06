#!/bin/bash
# One cohort after 04_union_db.py: PDC raw -> MGF -> reporters + Sage top-5 -> reduced outputs -> statistics.
# usage: run_cohort.sh <PDC study>, from the working directory holding data/ and results/
set -euo pipefail
STUDY=$1; HERE=$(dirname "$0"); PAN=results/cptac/pan
MGF=data/cptac_mgf/$STUDY; SAGE=$PAN/sage/$STUDY; COHORT=$PAN/cohorts/$STUDY
PROJECT=$(python -c "from juncta.config import COHORTS; print(COHORTS['$STUDY'][1])")
mkdir -p "$COHORT" data/pdc
python "$HERE/05_fetch.py" raw "$STUDY"
python "$HERE/05_fetch.py" channels "$STUDY"
python "$HERE/06_search.py" "$MGF" $PAN/db_v1/db.fasta "$SAGE" "$COHORT/reporter.parquet"
cp "$SAGE/cfg.json" "$SAGE/results.json" "$COHORT/"
python "$HERE/07_reduce.py" "$SAGE/results.sage.parquet" "$MGF" "$COHORT" "$COHORT/reporter.parquet" \
  "$PAN/spectra/$STUDY"
python "$HERE/08_cohort_stats.py" "$COHORT" $PAN/db_v1 "$PROJECT" "$PAN/analysis/$STUDY"
