#!/bin/bash

export FAIRSEQ_ROOT=<PATH_TO_FAIRSEQ_ROOT>
export VENV_ROOT=<PATH_TO_VENV>
export MUSTC_ROOT=<PATH_TO_MUSTC_DATA>
export LANGUAGE_PAIR=<LANGUAGE_PAIR_OF_INTEREST>
export ASR_SAVE_DIR=<PATH_TO_ASR_SAVE_DIR>

# activation of environment, moving to working directory
cd ${FAIRSEQ_ROOT}
source ${VENV_ROOT}/bin/activate

# added due to issues on our end with scoping in fairseq, can probably be removed
export PYTHONPATH='${PYTHONPATH}:.'


# execute ASR pre-training with some assumed hyperparameters based on our tests
fairseq-train ${MUSTC_ROOT}/${LANGUAGE_PAIR} \
    --config-yaml config_asr.yaml --train-subset train_asr --valid-subset dev_asr \
    --task speech_to_text --criterion label_smoothed_cross_entropy --report-accuracy \
    --save-dir ${ASR_SAVE_DIR} --num-workers 2 --max-tokens 30000 --max-update 200000 --max-epoch 40 \
    --arch convtransformer_espnet --optimizer adam --adam-betas [0.9,0.98] --lr 0.00025 --lr-scheduler inverse_sqrt \
    --warmup-updates 8000 --warmup-init-lr 0.0001 --clip-norm 10.0 --seed 1 --update-freq 2 \
    --ddp-backend legacy_ddp \
    --log-interval 50 \
    --encoder-normalize-before --decoder-normalize-before --share-decoder-input-output-embed \
    --attention-dropout 0.1 --activation-dropout 0.1 \
    --encoder-attention-heads 8 --decoder-attention-heads 8 \
    --encoder-leapformer-enable \
    --reset-optimizer \