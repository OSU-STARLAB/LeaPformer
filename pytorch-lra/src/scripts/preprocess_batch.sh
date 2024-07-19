#!/bin/bash

cd <PATH_TO_SRC>
source <VENV_ROOT>

export _PATHFINER_TFDS_PATH=<PATH_TO_TFDS>

mkdir preprocess
cd preprocess

python create_listops.py
python create_retrieval.py
python create_text.py
python create_pathfinder.py
python create_cifar10.py
