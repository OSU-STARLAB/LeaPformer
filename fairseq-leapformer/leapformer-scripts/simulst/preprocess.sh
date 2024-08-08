#!/bin/bash

export FAIRSEQ_ROOT=<PATH_TO_FAIRSEQ_ROOT>
export VENV_ROOT=<PATH_TO_VENV>
export MUSTC_ROOT=<PATH_TO_MUSTC_DATA>

# this example is for MuST-C, but CoVoST is also supported, although
# we recommend resampling to 16kHz before running this script for CoVoST,
# which matches the MuST-C sampling rate (CoVoST is 48kHz by default)
# e.g.  export COVOST_ROOT=<PATH_TO_COVOST_DATA>

# workaround for edge case issue with fairseq
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd ${FAIRSEQ_ROOT}
source ${VENV_ROOT}/bin/activate


# assumes that tar ball of MuST-C data is already placed where it should be and unzipped
# we are assuming en-de as the language pair of interest, CoVoST is fr-en
python ${FAIRSEQ_ROOT_LIN}/examples/speech_to_text/prep_mustc_data.py \
      --data-root ${FAIRSEQ_ROOT}/${MUSTC_ROOT} --task asr  --langs-to-process de \
      --vocab-type unigram --vocab-size 10000 \
      --cmvn-type global

python ${FAIRSEQ_ROOT_LIN}/examples/speech_to_text/prep_mustc_data.py \
      --data-root ${FAIRSEQ_ROOT}/${MUSTC_ROOT} --task st  --langs-to-process de \
      --vocab-type unigram --vocab-size 10000 \
      --cmvn-type global

#python ${FAIRSEQ_ROOT_LIN}/examples/speech_to_text/prep_covost_data.py \
#      --data-root ${FAIRSEQ_ROOT}/${COVOST_ROOT} --src-lang fr \
#      --vocab-type unigram --vocab-size 10000 \
#      --cmvn-type global \
#
#python ${FAIRSEQ_ROOT_LIN}/examples/speech_to_text/prep_covost_data.py \
#      --data-root ${FAIRSEQ_ROOT_LIN}/${COVOST_ROOT} --src-lang fr --tgt-lang en \
#      --vocab-type unigram --vocab-size 10000 \
#      --cmvn-type global \
