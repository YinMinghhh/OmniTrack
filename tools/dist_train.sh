#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-28650}

PYTHONPATH="$(dirname $0)/..:$(dirname $0)/../jrdb_toolkit/detection_eval":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
