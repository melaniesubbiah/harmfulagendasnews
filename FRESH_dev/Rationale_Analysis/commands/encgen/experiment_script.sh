echo ${CLASSIFIER:?"Set a Classifier"}

export EXP_NAME=$MAX_LENGTH_RATIO/$EXP_NAME;

bash Rationale_Analysis/commands/model_a_train_script.sh;

for rationale in top_k contiguous;
    do
    RATIONALE=$rationale \
    RATIONALE_EXP_NAME=$MAX_LENGTH_RATIO \
    bash Rationale_Analysis/commands/encgen/predict_script.sh;
    done;