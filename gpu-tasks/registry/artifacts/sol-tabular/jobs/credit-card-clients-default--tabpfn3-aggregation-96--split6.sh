#!/bin/sh
set -eu
exec python tools/single_cell.py --dataset credit_card_clients_default --split 6 --arm tabpfn3_aggregation_96 --output results/credit-card-clients-default--tabpfn3-aggregation-96--split6.json
