#!/bin/bash

export FAIRSEQ_ROOT=<PATH_TO_FAIRSEQ_ROOT>
export VENV_ROOT=<PATH_TO_VENV>
export WIKITEXT_SRC=<PATH_TO_WIKITEXT_BASIC_DATA>
export WIKITEXT_DEST=<PATH_TO_WIKITEXT_DEST>

# activation of environment, moving to working directory
cd ${FAIRSEQ_ROOT}
source ${VENV_ROOT}/bin/activate

# added due to issues on our end with scoping in fairseq, can probably be removed
export PYTHONPATH='${PYTHONPATH}:.'

fairseq-preprocess \
    --only-source \
    --trainpref ${WIKITEXT_SRC}/wiki.train.tokens \
    --validpref ${WIKITEXT_SRC}/wiki.valid.tokens \
    --testpref ${WIKITEXT_SRC}/wiki.test.tokens \
    --destdir ${WIKITEXT_DEST} \
    --workers 20 \
