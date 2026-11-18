#!/bin/sh
set -eu
exec python tools/single_cell.py --dataset physiochemical_protein --split 3 --arm tabpfn3_diagscale_aggregation_96 --output results/physiochemical-protein--tabpfn3-diagscale-aggregation-96--split3.json
