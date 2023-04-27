# Detecting Harmful Agendas in News Articles
Code and data for the paper - [Detecting Harmful Agendas in News Articles](https://arxiv.org/abs/2302.00102).

## NewsAgendas Dataset
The annotated data can be found in the file *newsagendas.jsonl*.
- **id**: Article id.
- **article-title**: Title of the article.
- **article-contents**: Cleaned/formatted article contents.
- **annotated-labels**: Annotated feature labels.
  - clickbait
  - junkscience
  - hatespeech
  - conspiracytheory
  - propaganda
  - satire
  - negativesentiment
  - neutralsentiment
  - positivesentiment
  - politicalbias
  - calltoaction
- **annotated-agenda-score**: Annotated agenda score on a scale 1 to 5 with 1 being clearly benign and 5 being clearly malicious. The value is \'no answer\' if the annotator did not assign a score.
- **annotated-evidence**: Snippets of text highlighted by the annotators as evidence for the feature labels they annotated. These snippets are copied directly from the article and formatted as a dictionary.
- **split**: Which split (dev or test) the article is assigned to (necessary to replicate results from the paper). Articles without an agenda score are assigned to the 'full' split.
- **weak-label-0**: Original source-level label assigned to the article. The first one listed by the [FakeNewsCorpus](https://github.com/several27/FakeNewsCorpus).
- **weak-label-1**: Original source-level label assigned to the article. The second one listed by the [FakeNewsCorpus](https://github.com/several27/FakeNewsCorpus).
- **weak-label-2**: Original source-level label assigned to the article. The third one listed by the [FakeNewsCorpus](https://github.com/several27/FakeNewsCorpus).

## Evaluation Code
The results shown in the paper were generated using *Results_Tables.ipynb*.

## Finetuning BERT model
To finetune a BERT model to predict the agenda score from the article title and contents, we use the data splits found in *bert_training_datasets* for training with cross-validation. You can finetune BERT on these splits to replicate our results in the paper by running:
```
python BERT_model.py
```
