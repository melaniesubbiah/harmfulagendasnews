1. 
BERT models' predictions are in *_bert_predictions.json
FRESH models' predictions are in *_fresh_predictions.json

2. 
In any predictions.json file, typical entries look like the following:
{"predicted_label": "1", "label": "1", "annotation_id": "old0_0"}
{"predicted_label": "0", "label": "1", "annotation_id": "full_agreement54_617"}
{"predicted_label": "0", "label": "1", "annotation_id": "0_643"}

Notice that "label": "1" is a placeholder. This item will always be 1 for all entries. 

3.
The "annotation_id" is used to match indices across files.

