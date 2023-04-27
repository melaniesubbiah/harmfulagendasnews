export CONFIG_FILE=Rationale_Analysis/training_config/classifiers/${CLASSIFIER:?"Set classifier"}.jsonnet
export CUSTOMIZED_LABEL_WEIGHTS=$CUSTOMIZED_LABEL_WEIGHTS
export CUDA_DEVICE=${CUDA_DEVICE:?"set cuda device"}

export TRAIN_DATA_PATH=${DATA_BASE_PATH:?"set data base path"}/train.jsonl
export DEV_DATA_PATH=$DATA_BASE_PATH/dev.jsonl
export TEST_DATA_PATH=$DATA_BASE_PATH/test.jsonl
export EVAL_DATA_PATH=$DATA_BASE_PATH/human_eval.jsonl

export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs}/${DATASET_NAME:?"Set dataset name"}/$CLASSIFIER/${EXP_NAME:?"Set Exp name"}

export SEED=${RANDOM_SEED:-100}

if [[ -f "${OUTPUT_BASE_PATH}/metrics.json" && -z "$again" ]]; then
    echo "${OUTPUT_BASE_PATH}/metrics.json exists ... . Not running Training ";
else 
    echo "${OUTPUT_BASE_PATH}/metrics.json does not exist ... . TRAINING ";
    allennlp train -s $OUTPUT_BASE_PATH --include-package Rationale_Analysis --force $CONFIG_FILE
fi




if [[ -f "${OUTPUT_BASE_PATH}/predictions.jsonl" && -z "$again" ]]; then
    echo "${OUTPUT_BASE_PATH}/predictions.jsonl exists ... . Not running predicting ";
else
    echo "${OUTPUT_BASE_PATH}/predictions.jsonl does not exist ... . PREDICTING ";
    allennlp predict $OUTPUT_BASE_PATH/model.tar.gz $EVAL_DATA_PATH \
    --output-file $OUTPUT_BASE_PATH/predictions.jsonl \
    --use-dataset-reader \
    --cuda-device $CUDA_DEVICE \
    --dataset-reader-choice train \
    --include-package Rationale_Analysis \
    --predictor rationale_predictor
fi