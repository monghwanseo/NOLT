#!/bin/bash
set -e
PY=${PYTHON:-python}
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
mkdir -p paper/tables paper/figures/pdf

for f in code/01_data/0*.py; do
    $PY "$f"
done

$PY code/02_experiments/05_factor_ladder.py
$PY code/02_experiments/09_vol_benchmark_family.py

for n in 01 02 03 04 06 07 08 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31; do
    f=$(ls code/02_experiments/${n}_*.py)
    $PY "$f"
done
