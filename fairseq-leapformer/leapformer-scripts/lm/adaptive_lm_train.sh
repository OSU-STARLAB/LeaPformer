#!/bin/bash

export FAIRSEQ_ROOT=<PATH_TO_FAIRSEQ_ROOT>
export VENV_ROOT=<PATH_TO_VENV>
export WIKITEXT_ROOT=<PATH_TO_WIKITEXT_DATA>
export LM_SAVE_DIR=<PATH_TO_LM_SAVE_DIR>

# activation of environment, moving to working directory
cd ${FAIRSEQ_ROOT}
source ${VENV_ROOT}/bin/activate

# added due to issues on our end with scoping in fairseq, can probably be removed
export PYTHONPATH='${PYTHONPATH}:.'


fairseq-train ${WIKITEXT_ROOT} \
    --task language_modeling \
    --save-dir ${LM_SAVE_DIR} \
    --arch transformer_lm_wiki103 \
    --max-update 250000 --lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
    --warmup-updates 8000 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --optimizer nag --min-lr 0.0001 --clip-norm 0.1 \
    --criterion adaptive_loss --max-tokens 4096 --update-freq 2 --tokens-per-sample 512 --seed 1 \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=legacy_ddp \
    --dec-leapformer-enable \

fairseq-eval-lm ${WIKITEXT_ROOT} \
    --path ${LM_SAVE_DIR}/checkpoint_best.pt \
    --batch-size 2 \
    --tokens-per-sample 512 \
    --context-window 511 \
