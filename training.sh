#!/bin/bash

python3 train.py --encoder-type resnet18 --decoder-type transformer --num-heads 1 --num-tf-layers 3  --experiment-name resnet18_l3_h1
python3 train.py --encoder-type resnet18 --decoder-type transformer --num-heads 2 --num-tf-layers 3  --experiment-name resnet18_l3_h2
python3 train.py --encoder-type resnet18 --decoder-type transformer --num-heads 3 --num-tf-layers 3  --experiment-name resnet18_l3_h3
python3 train.py --encoder-type resnet18 --decoder-type transformer --num-heads 1 --num-tf-layers 5  --experiment-name resnet18_l5_h1
python3 train.py --encoder-type resnet18 --decoder-type transformer --num-heads 2 --num-tf-layers 5  --experiment-name resnet18_l5_h2
python3 train.py --encoder-type resnet18 --decoder-type transformer --num-heads 3 --num-tf-layers 5  --experiment-name resnet18_l5_h3
python3 train.py --encoder-type resnet18 --decoder-type transformer --num-heads 1 --num-tf-layers 7  --experiment-name resnet18_l7_h1
python3 train.py --encoder-type resnet18 --decoder-type transformer --num-heads 2 --num-tf-layers 7  --experiment-name resnet18_l7_h2
python3 train.py --encoder-type resnet18 --decoder-type transformer --num-heads 3 --num-tf-layers 7  --experiment-name resnet18_l7_h3