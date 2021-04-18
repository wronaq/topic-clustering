#!/bin/sh

if [ $# -lt 1 ]; then
    echo "Expecting "
    echo "Expecting: "
    echo "1) path to config file,"
    echo "2) number of topics to find. If zero then no reduction is made,"
    echo "3) top n words to describe each topic,"
    echo "4) flag --save_embeddings (optional)"
    exit 1
fi


export PYTHONPATH=$(pwd)
python run/experiment.py $1 $2

