#!/bin/sh
set -eu
exec python tools/single_cell.py --dataset customer_satisfaction_in_airline --split 6 --arm tabpfn3_aggregation_96 --output results/customer-satisfaction-in-airline--tabpfn3-aggregation-96--split6.json
