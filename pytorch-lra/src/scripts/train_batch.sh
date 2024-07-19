#!/bin/bash

cd <PATH_TO_SRC>
source <VENV_ROOT>

python main.py \
    --mode train --attn leapformer --task lra-text --log-affix leapformer-cls \
    --pooling-mode CLS \
    --learned-numer-inter-size 1 \
