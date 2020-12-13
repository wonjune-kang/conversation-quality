import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset


class FullMetricDataset(Dataset):
    def __init__(self, data_csv, labels_csv, xval_idx=0, is_train=True,
                 ablation_feat=None, drop_feats=["Conversation ID", "Mean inter-speaker silence", "Interruption rate"]):
        if ablation_feat is not None:
            drop_feats.append(ablation_feat)

        data_df = pd.read_csv(data_csv, delimiter='\t').drop(drop_feats, axis=1)
        self.data_np = data_df.to_numpy()
        self.data_scaler = StandardScaler()
        self.data_scaled = self.data_scaler.fit_transform(self.data_np)

        labels_df = pd.read_csv(labels_csv).drop(["treatment", "conversation"], axis=1)
        self.labels_np = labels_df.to_numpy()
        self.labels_scaler = StandardScaler()
        self.labels_scaled = self.labels_scaler.fit_transform(self.labels_np)

        if xval_idx > 9:
            raise Exception("Invalid cross validation index for Keeper dataset")

        if is_train:
            if xval_idx != 9:
                self.data_scaled = np.concatenate([self.data_scaled[:2*xval_idx],
                                                   self.data_scaled[2*(xval_idx+1):]], axis=0)
                self.labels_scaled = np.concatenate([self.labels_scaled[:2*xval_idx],
                                                     self.labels_scaled[2*(xval_idx+1):]], axis=0)
            elif xval_idx == 9:
                self.data_scaled = self.data_scaled[:2*xval_idx]
                self.labels_scaled = self.labels_scaled[:2*xval_idx]
        else:
            if xval_idx != 9:
                self.data_scaled = self.data_scaled[2*xval_idx:2*(xval_idx+1)]
                self.labels_scaled = self.labels_scaled[2*xval_idx:2*(xval_idx+1)]
            elif xval_idx == 9:
                self.data_scaled = self.data_scaled[2*xval_idx:]
                self.labels_scaled = self.labels_scaled[2*xval_idx:]

    def __len__(self):
        return len(self.data_scaled)

    def __getitem__(self, i):
        return torch.tensor(self.data_scaled[i], dtype=torch.float), torch.tensor(self.labels_scaled[i], dtype=torch.float)


class SingleMetricDataset(Dataset):
    def __init__(self, data_csv, labels_csv, pred_val, xval_idx=0,
                 is_train=True, ablation_feat=None,
                 drop_feats=["Conversation ID", "Mean inter-speaker silence", "Interruption rate"]):
        if ablation_feat is not None:
            drop_feats.append(ablation_feat)

        data_df = pd.read_csv(data_csv, delimiter='\t').drop(drop_feats, axis=1)
        self.data_np = data_df.to_numpy()
        self.data_scaler = StandardScaler()
        self.data_scaled = self.data_scaler.fit_transform(self.data_np)

        labels_df = pd.read_csv(labels_csv)[pred_val]
        self.labels_np = labels_df.to_numpy().reshape(-1, 1)
        self.labels_scaler = StandardScaler()
        self.labels_scaled = self.labels_scaler.fit_transform(self.labels_np)

        if is_train:
            if xval_idx != 9:
                self.data_scaled = np.concatenate([self.data_scaled[:2*xval_idx],
                                                   self.data_scaled[2*(xval_idx+1):]], axis=0)
                self.labels_scaled = np.concatenate([self.labels_scaled[:2*xval_idx],
                                                     self.labels_scaled[2*(xval_idx+1):]], axis=0)
            elif xval_idx == 9:
                self.data_scaled = self.data_scaled[:2*xval_idx]
                self.labels_scaled = self.labels_scaled[:2*xval_idx]
        else:
            if xval_idx != 9:
                self.data_scaled = self.data_scaled[2*xval_idx:2*(xval_idx+1)]
                self.labels_scaled = self.labels_scaled[2*xval_idx:2*(xval_idx+1)]
            elif xval_idx == 9:
                self.data_scaled = self.data_scaled[2*xval_idx:]
                self.labels_scaled = self.labels_scaled[2*xval_idx:]

    def __len__(self):
        return len(self.data_scaled)

    def __getitem__(self, i):
        return torch.tensor(self.data_scaled[i], dtype=torch.float), torch.tensor(self.labels_scaled[i], dtype=torch.float)




