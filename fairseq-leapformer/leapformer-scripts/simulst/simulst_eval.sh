#!/bin/bash

export FAIRSEQ_ROOT=<PATH_TO_FAIRSEQ_ROOT>
export VENV_ROOT=<PATH_TO_VENV>
export MUSTC_ROOT=<PATH_TO_MUSTC_DATA>
export LANGUAGE_PAIR=<LANGUAGE_PAIR_OF_INTEREST>
export ST_SAVE_DIR=<PATH_TO_ST_SAVE_DIR>

export SIMUL_EVAL_ROOT=<PATH_TO_SIMUL_EVAL>
export EVAL_DATA=<PATH_TO_POST_SPLIT_EVAL_DATA>

export WAV_LIST=<NAME_OF_WAV_LIST>
export TARGETS=<NAME_OF_TARGET_TRANSLATION_FILE>

# activation of environment, moving to working directory
cd ${FAIRSEQ_ROOT}
source ${VENV_ROOT}/bin/activate

# added due to issues on our end with scoping in fairseq, can probably be removed
export PYTHONPATH='${PYTHONPATH}:.'

cd ${SIMUL_EVAL_ROOT}

simuleval \
    --agent ${FAIRSEQ_ROOT}/examples/speech_to_text/simultaneous_translation/agents/fairseq_simul_st_agent.py \
    --data-bin ${MUSTC_ROOT}/${LANGUAGE_PAIR} \
    --source ${EVAL_DATA}/${WAV_LIST} \
    --target ${EVAL_DATA}/${TARGETS} \
    --config config_st.yaml \
    --model-path ${ST_SAVE_DIR}/checkpoint_best.pt \
    --output test-output \
    --waitk 5 \
    --scores \
    --force-finish \

