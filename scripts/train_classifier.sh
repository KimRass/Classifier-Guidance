#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

python3 ../train_classifier.py\
    --data_dir="/Users/jongbeomkim/Documents/datasets/"\
    --save_dir="/Users/jongbeomkim/Documents/classifier-guidance"\
    --n_epochs=50\
    --batch_size=8\
    --lr=0.0003\
    --n_cpus=2\
    --n_warmup_steps=1000\
