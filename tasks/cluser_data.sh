#!/bin/sh

if [ $# -lt 1 ]; then
    echo "Expecting "
    echo "Expecting: "
    echo "1) path to config file,"
    echo "2) flag --save_embeddings (optional)"
    exit 1
fi


export PYTHONPATH=$(pwd)
python run/experiment.py $1 $2

