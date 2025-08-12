#!/bin/bash


GPU=0
DATASET=upmc_food101_cmml
TYPE=both
RATIO=0.7
EXP_NOTE="${DATASET}_${TYPE}_${RATIO}_v27"

TOKENIZERS_PARALLELISM=True python src/main.py \
    experiment=rebq_mixlora_${DATASET} \
    data=${DATASET} \
    EXP_NOTE=${EXP_NOTE} \
    train.GPU=${GPU} \
    data.missing_params.RATIO=${RATIO} \
    data.missing_params.TYPE=${TYPE} \
    train.EVAL_FREQ=1000 \
    data.NUM_WORKERS=2 \
    test.BATCH_SIZE=4

TOKENIZERS_PARALLELISM=True python src/main.py experiment=rebq_mixlora_${DATASET} data=${DATASET} EXP_NOTE=${EXP_NOTE} train.GPU=${GPU} data.missing_params.RATIO=${RATIO} data.missing_params.TYPE=${TYPE} train.EVAL_FREQ=1000 data.NUM_WORKERS=2 test.BATCH_SIZE=4 test.TEST_ONLY=True test.CHECKPOINT_DIR=/l/users/xiwei.liu/5/RebQ.update/checkpoints/${EXP_NOTE}/checkpoints/
