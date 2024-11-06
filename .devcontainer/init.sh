#!/bin/bash

setup_new_user 1000 1000
git config --global --add safe.directory "*"

source /root/miniconda3/etc/profile.d/conda.sh

conda create -n rltorch \
  pytorch torchvision torchaudio \
  pytorch-cuda=12.1 gymnasium pyglet \
  pygame gymnasium-box2d colorama \
  pylint yapf tqdm 'tensorboardx>=2.5.0' \
  'tensorboard>2.0' pillow matplotlib scipy \
  seaborn ipykernel -c conda-forge -c pytorch -c nvidia
