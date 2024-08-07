#!/bin/bash

export FAIRSEQ_ROOT=<PATH_TO_FAIRSEQ_ROOT>
export VENV_ROOT=<PATH_TO_VENV>
export MUSTC_ROOT=<PATH_TO_MUSTC_DATA>
export LANGUAGE_PAIR=<LANGUAGE_PAIR_OF_INTEREST>
export ST_SAVE_DIR=<PATH_TO_ST_SAVE_DIR>

# activation of environment, moving to working directory
cd ${FAIRSEQ_ROOT}
source ${VENV_ROOT}/bin/activate

# added due to issues on our end with scoping in fairseq, can probably be removed
export PYTHONPATH='${PYTHONPATH}:.'


# execute SimulST training with some assumed hyperparameters based on our tests
fairseq-train ${MUSTC_ROOT_LIN}/en-zh \
    --task speech_to_text --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
    --load-pretrained-encoder-from ${ASR_SAVE_DIR_LIN}/checkpoint_best.pt \
    --save-dir ${ST_SAVE_DIR_LIN} \
    --arch convtransformer_simul_trans_espnet \
    --simul-type waitk_fixed_pre_decision --criterion label_smoothed_cross_entropy --fixed-pre-decision-ratio 9 --waitk-lagging 5 \
    --max-tokens 45000 --num-workers 2 --update-freq 2 --max-epoch 40 \
    --optimizer adam --adam-betas [0.9,0.98] --lr 0.00033 --lr-scheduler inverse_sqrt --warmup-updates 6000 --warmup-init-lr 0.0001 --clip-norm 10.0 \
    --max-update 100000 --seed 2 --ddp-backend legacy_ddp --log-interval 50 \
    --encoder-normalize-before --decoder-normalize-before --share-decoder-input-output-embed \
    --encoder-attention-heads 8 --decoder-attention-heads 8 \
    --encoder-leapformer-enable \
    --decoder-sa-leapformer-enable \
    --decoder-ca-leapformer-enable \
    --reset-optimizer \
