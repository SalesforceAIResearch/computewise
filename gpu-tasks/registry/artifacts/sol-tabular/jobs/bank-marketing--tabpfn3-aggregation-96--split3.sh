#!/bin/sh
set -eu
exec python tools/single_cell.py --dataset bank-marketing --split 3 --arm tabpfn3_aggregation_96 --output results/bank-marketing--tabpfn3-aggregation-96--split3.json
