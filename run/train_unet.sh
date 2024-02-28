#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

python3 ../train_unet.py\
    --classifier_params="/Users/jongbeomkim/Documents/classifier-guidance/epoch=4-val_loss=0.0000.pth"\
    --data_dir="/Users/jongbeomkim/Documents/datasets/"\
    --save_dir="/Users/jongbeomkim/Documents/classifier-guidance"\
    --n_epochs=30\
    --batch_size=128\
    --lr=0.0003\
    --n_cpus=3\
    --n_warmup_steps=500\
