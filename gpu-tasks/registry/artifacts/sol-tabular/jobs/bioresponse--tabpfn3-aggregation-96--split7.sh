#!/bin/sh
set -eu
exec python tools/single_cell.py --dataset Bioresponse --split 7 --arm tabpfn3_aggregation_96 --output results/bioresponse--tabpfn3-aggregation-96--split7.json
