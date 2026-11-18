#!/bin/sh
set -eu
exec python tools/single_cell.py --dataset QSAR-TID-11 --split 6 --arm tabpfn3_diagscale_aggregation_96 --output results/qsar-tid-11--tabpfn3-diagscale-aggregation-96--split6.json
