#!/bin/bash

export FAIRSEQ_ROOT=<PATH_TO_FAIRSEQ_ROOT>
export VENV_ROOT=<PATH_TO_VENV>
export MUSTC_ROOT=<PATH_TO_MUSTC_DATA>
export ST_SAVE_DIR=<PATH_TO_ST_SAVE_DIR>

export SIMUL_EVAL_ROOT=<PATH_TO_SIMUL_EVAL>
export EVAL_DATA=<PATH_TO_POST_SPLIT_EVAL_DATA>

# choices are usually dev, tst-HE, or tst-COMMON
# tst-COMMON should be used by default
export SPLIT=<DATA_SPLIT_OF_CHOICE>

# this example is for MuST-C, but CoVoST is also supported, although
# we recommend resampling to 16kHz before running this script for CoVoST,
# which matches the MuST-C sampling rate (CoVoST is 48kHz by default)
# e.g.  export COVOST_ROOT=<PATH_TO_COVOST_DATA>

# workaround for edge case issue with fairseq
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# activation of environment, moving to working directory
cd ${FAIRSEQ_ROOT}
source ${VENV_ROOT}/bin/activate

# added due to issues on our end with scoping in fairseq, can probably be removed
export PYTHONPATH='${PYTHONPATH}:.'

python ${FAIRSEQ_ROOT}/examples/speech_to_text/seg_mustc_data.py \
  --data-root ${MUSTC_ROOT} --lang de \
  --split ${SPLIT} --task st \
  --output ${EVAL_DATA}

#python ${FAIRSEQ_ROOT}/examples/speech_to_text/seg_covost_data.py \
#    --data-root ${COVOST_ROOT_FR} --src-lang fr --tgt-lang en \
#    --output ${EVAL_DATA_FR} \
#    --split test --task st \
#    #--data-root ${COVOST_ROOT_FR} --lang fr \
#    #--split ${SPLIT} --task st \
