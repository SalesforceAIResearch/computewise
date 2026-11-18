#!/bin/sh
set -eu
exec python tools/single_cell.py --dataset Bioresponse --split 4 --arm tabpfn3_diagscale_aggregation_96 --output results/bioresponse--tabpfn3-diagscale-aggregation-96--split4.json
