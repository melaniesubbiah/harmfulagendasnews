bash Rationale_Analysis/commands/model_a_train_script.sh;

export PYTHONPATH=.

#for saliency in wrapper simple_gradient;
for saliency in wrapper;
    do 
    SALIENCY=$saliency bash Rationale_Analysis/commands/fresh/saliency_script.sh;
#    for thresholder in top_k contiguous;
    for thresholder in top_k;
        do
        SALIENCY=$saliency \
        THRESHOLDER=$thresholder \
        THRESHOLDER_EXP_NAME=$MAX_LENGTH_RATIO \
        bash Rationale_Analysis/commands/fresh/thresholder_and_model_b_script.sh;
        done;
    done;