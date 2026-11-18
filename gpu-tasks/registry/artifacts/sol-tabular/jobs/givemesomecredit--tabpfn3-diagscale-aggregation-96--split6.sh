#!/bin/sh
set -eu
exec python tools/single_cell.py --dataset GiveMeSomeCredit --split 6 --arm tabpfn3_diagscale_aggregation_96 --output results/givemesomecredit--tabpfn3-diagscale-aggregation-96--split6.json
