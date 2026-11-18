#!/bin/sh
set -eu
exec python tools/single_cell.py --dataset bank-marketing --split 1 --arm tabpfn3_diagscale_aggregation_96 --output results/bank-marketing--tabpfn3-diagscale-aggregation-96--split1.json
