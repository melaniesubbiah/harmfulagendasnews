Our modifications have been incorporated into the original pipeline of the FRESH framework. For the required packages and more complete instructions, take a look at the README.md made by the authors of FRESH. 



Place the unzipped dataset folder under `FRESH_dev/datasets`

Three example datasets have been uploaded to https://drive.google.com/drive/folders/1a9XXaEFslcFg0aqKw078jaHN75cEO8e0?usp=sharing





Running the toy dataset

```
CUDA_DEVICE=0 \
DATASET_NAME=toy \
CLASSIFIER=bert_classification \
BERT_TYPE=bert-base-cased \
EXP_NAME=fresh \
MAX_LENGTH_RATIO=0.2 \
SALIENCY=wrapper \
THRESHOLDER=top_k \
EPOCHS=20 \
BSIZE=16 \
bash Rationale_Analysis/commands/fresh/fresh_script.sh

```

* other configurations can be changed in `FRESH_dev/Rationale_Analysis/training_config/classifiers/bert_classification.jsonnet`



Running across different random seeds

```
CUDA_DEVICE=1 \
EPOCHS=50 \
DATASET_NAME=7class \
CLASSIFIER=bert_classification \
python Rationale_Analysis/experiments/run_for_random_seeds.py \
--script-type fresh/experiment_script.sh \
--defaults-file Rationale_Analysis/default_values/news_b16_r0.2.json
```





# Results

`FRESH_dev/extracted_results`



BERT results are in the first predictions.json `FRESH_dev/extracted_results/detect_clickbait/bert_classification/random_seed_variance/RANDOM_SEED=1000/predictions.json`

![image-20220110165048309](D:\Work\nlp_research_local\characterization-components\FRESH_dev\README_by_Bobby.assets\image-20220110165048309.png)



Rationale model's predicted labels are in the second predictions.json `FRESH_dev/extracted_results/detect_clickbait/bert_classification/random_seed_variance/RANDOM_SEED=1000/wrapper_saliency/top_k_thresholder/0.2/model_b/predictions.json`

![image-20220110165139784](D:\Work\nlp_research_local\characterization-components\FRESH_dev\README_by_Bobby.assets\image-20220110165139784.png)



The extracted rationales are in `FRESH_dev/extracted_results/detect_clickbait/bert_classification/random_seed_variance/RANDOM_SEED=1000/wrapper_saliency/top_k_thresholder/0.2/human_eval.jsonl`

![image-20220110165500674](D:\Work\nlp_research_local\characterization-components\FRESH_dev\README_by_Bobby.assets\image-20220110165500674.png)





