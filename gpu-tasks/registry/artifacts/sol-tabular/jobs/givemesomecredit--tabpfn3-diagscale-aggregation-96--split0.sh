#!/bin/sh
set -eu
exec python tools/single_cell.py --dataset GiveMeSomeCredit --split 0 --arm tabpfn3_diagscale_aggregation_96 --output results/givemesomecredit--tabpfn3-diagscale-aggregation-96--split0.json
