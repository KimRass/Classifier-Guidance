#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

save_dir="/Users/jongbeomkim/Desktop/workspace/Classifier-Guidance/samples"
img_size=32
classifier_scale=20

python3 ../sample.py\
    --model_params="/Users/jongbeomkim/Documents/classifier-guidance/unet_channels=128/epoch=28-val_loss=0.0303.pth"\
    --classifier_scale=$classifier_scale\
    --batch_size=4\
    --img_size=$img_size
