#!/bin/sh
set -eu
exec python tools/single_cell.py --dataset Bioresponse --split 5 --arm tabpfn3_aggregation_96 --output results/bioresponse--tabpfn3-aggregation-96--split5.json
