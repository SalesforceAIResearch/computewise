#!/bin/sh
set -eu
exec python tools/single_cell.py --dataset GiveMeSomeCredit --split 2 --arm tabpfn3_aggregation_96 --output results/givemesomecredit--tabpfn3-aggregation-96--split2.json
