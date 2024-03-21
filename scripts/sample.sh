#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

img_size=32
classifier_scale=200

python3 ../sample.py\
    --model_params="/home/dmeta0304/Downloads/classifier_guidance-cifar10.pth"\
    --classifier_scale=$classifier_scale\
    --batch_size=10\
    --img_size=$img_size
