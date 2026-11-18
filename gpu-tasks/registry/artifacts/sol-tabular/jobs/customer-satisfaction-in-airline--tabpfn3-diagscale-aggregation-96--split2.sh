#!/bin/sh
set -eu
exec python tools/single_cell.py --dataset customer_satisfaction_in_airline --split 2 --arm tabpfn3_diagscale_aggregation_96 --output results/customer-satisfaction-in-airline--tabpfn3-diagscale-aggregation-96--split2.json
