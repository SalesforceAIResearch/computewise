#!/bin/sh
set -eu
exec python tools/single_cell.py --dataset Bioresponse --split 1 --arm tabpfn3_aggregation_96 --output results/bioresponse--tabpfn3-aggregation-96--split1.json
