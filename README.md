# conversation-quality

This repository contains code for a class project for the Understanding
Public Thought class at MIT (MAS.S62), Fall 2020.

We compute several lexical metrics from transcriptions of spoken conversations
and use them as conditioning features to try to estimate conversation quality.
Special thanks to Doug Beeferman for providing the baseline code for
`utils/lexical_metrics.py` and `utils/responsivity.py`.


## Dependencies

* python==3.7+
* numpy==1.19.4
* scipy==1.5.4
* scikit-learn==0.23.2
* pandas==1.1.3
* spacy==2.3.4
* textatistic==0.0.1
* vaderSentiment==3.3.2
* torch==1.7.0
* sentence-transformers==0.3.9

Install all dependencies by running

```
pip install requirements.txt
```

spacy, textstatistic, vaderSentiment, and sentence-transformers not needed to
train the model, but are required to pre-extract features before training.


## Data

Conversation data used for training and evaluation can be found in `data/json`.
Speaker turn information is saved in the following format:

```
[
    {
        "audio_start_offset": 0.0,
        "audio_end_offset": 10.0,
        "speaker_id": "Speaker 1",
        "content": "This is a sample sentence."
    }
    ...
]
```

Pre-extracted features used for the analysis can be found in `data/csv`.
These CSV files can be generated from scratch by running `extract_metrics.py`.
For example, the following command computes metrics for all conversations in
`data/json/keeper_json_clean` and creates `data/csv/keeper_metrics_full.csv`:

```
python3 extract_metrics.py data/json/keeper_json_clean data/csv/keeper_metrics_full.csv
```

## Experiments

Run `train.py` to train a model. Possible model architectures are in
`models.py`. If training a model to predict a single label rather than multiple
labels simultaneously, the `--single_regression` flag must be specified.

Ablation studies may be run by specifying the `--ablation` flag and
specifying the feature to be ablated in the `--ablation_feat` argument.

By default, the scipt performs 5 training runs per cross validation fold
in order to account for the effects of weight initialization. This value can
be adjusted using the `--runs_per_fold` argument.

The following command will train a 4-way regression model.
Checkoints will be saved to `./checkpoints` and mean square error for
all cross validation folds will be saved to `mrn_keeper_full.csv`.

```
python3 train.py --data_csv ./data/csv/keeper_metrics_full.csv --labels_csv ./data/csv/keeper_survey_labels.csv --ckpt_path checkpoints --log_file mrn_keeper_full.csv
```

The saved model weights can then be loaded in and used to evaluate other
conversations. Note that lexical metrics must be pre-extracted and saved
before doing this. (e.g. `data/csv/lvn_metrics.csv`)


