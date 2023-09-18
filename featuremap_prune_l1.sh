#!/bin/bash

source seeds

for epoch in 100 200 300
do
    for sparsity in 0.3 0.5 0.7 0.9
    do
        python3 featuremap_prune.py \
                --datapath data/cwru \
                --model wdcnn \
                --seed $SEED \
                --epochs $epoch \
                --activation-drop featuremap \
                --activation-sparsity $sparsity \
                --score-type l1 \
                --snr-value 0 4 8 -4 -8 \
                --device 1
    done
done