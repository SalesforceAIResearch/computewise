#!/bin/sh
set -eu
exec python tools/single_cell.py --dataset APSFailure --split 0 --arm tabpfn3_aggregation_96 --output results/apsfailure--tabpfn3-aggregation-96--split0.json
