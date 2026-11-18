#!/bin/sh
set -eu
exec python tools/single_cell.py --dataset credit_card_clients_default --split 4 --arm tabpfn3_diagscale_aggregation_96 --output results/credit-card-clients-default--tabpfn3-diagscale-aggregation-96--split4.json
