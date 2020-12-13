#!/bin/bash

python3 train.py \
    --data_csv ./data/csv/keeper_metrics_full.csv \
    --labels_csv ./data/csv/keeper_survey_avg.csv \
    --ckpt_path ./checkpoints/mrn_keeper_full_abl_wph \
    --log_file ./logs/mrn_keeper_full_abl_wph.csv \
    --ablation True \
    --ablation_feat "Words per hour"

python3 train.py \
    --data_csv ./data/csv/keeper_metrics_full.csv \
    --labels_csv ./data/csv/keeper_survey_avg.csv \
    --ckpt_path ./checkpoints/mrn_keeper_full_abl_spt \
    --log_file ./logs/mrn_keeper_full_abl_spt.csv \
    --ablation True \
    --ablation_feat "Speech per turn"

python3 train.py \
    --data_csv ./data/csv/keeper_metrics_full.csv \
    --labels_csv ./data/csv/keeper_survey_avg.csv \
    --ckpt_path ./checkpoints/mrn_keeper_full_abl_ttb \
    --log_file ./logs/mrn_keeper_full_abl_ttb.csv \
    --ablation True \
    --ablation_feat "Turn taking balance"

python3 train.py \
    --data_csv ./data/csv/keeper_metrics_full.csv \
    --labels_csv ./data/csv/keeper_survey_avg.csv \
    --ckpt_path ./checkpoints/mrn_keeper_full_abl_gl \
    --log_file ./logs/mrn_keeper_full_abl_gl.csv \
    --ablation True \
    --ablation_feat "Grade level"

python3 train.py \
    --data_csv ./data/csv/keeper_metrics_full.csv \
    --labels_csv ./data/csv/keeper_survey_avg.csv \
    --ckpt_path ./checkpoints/mrn_keeper_full_abl_mattr \
    --log_file ./logs/mrn_keeper_full_abl_mattr.csv \
    --ablation True \
    --ablation_feat "MATTR lexical diversity"

python3 train.py \
    --data_csv ./data/csv/keeper_metrics_full.csv \
    --labels_csv ./data/csv/keeper_survey_avg.csv \
    --ckpt_path ./checkpoints/mrn_keeper_full_abl_mwl \
    --log_file ./logs/mrn_keeper_full_abl_mwl.csv \
    --ablation True \
    --ablation_feat "Mean word length"

python3 train.py \
    --data_csv ./data/csv/keeper_metrics_full.csv \
    --labels_csv ./data/csv/keeper_survey_avg.csv \
    --ckpt_path ./checkpoints/mrn_keeper_full_abl_vader \
    --log_file ./logs/mrn_keeper_full_abl_vader.csv \
    --ablation True \
    --ablation_feat "VADER sentiment"

python3 train.py \
    --data_csv ./data/csv/keeper_metrics_full.csv \
    --labels_csv ./data/csv/keeper_survey_avg.csv \
    --ckpt_path ./checkpoints/mrn_keeper_full_abl_resp \
    --log_file ./logs/mrn_keeper_full_abl_resp.csv \
    --ablation True \
    --ablation_feat "Responsivity rate"