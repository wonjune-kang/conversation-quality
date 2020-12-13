#!/bin/bash

python3 train.py \
    --data_csv ./data/csv/keeper_metrics_full.csv \
    --labels_csv ./data/csv/keeper_survey_avg.csv \
    --ckpt_path ./checkpoints/mrn_keeper_full \
    --log_file ./logs/mrn_keeper_full.csv