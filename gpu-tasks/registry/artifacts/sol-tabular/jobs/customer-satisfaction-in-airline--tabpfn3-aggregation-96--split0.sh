#!/bin/sh
set -eu
exec python tools/single_cell.py --dataset customer_satisfaction_in_airline --split 0 --arm tabpfn3_aggregation_96 --output results/customer-satisfaction-in-airline--tabpfn3-aggregation-96--split0.json
