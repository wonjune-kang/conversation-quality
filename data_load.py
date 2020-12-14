import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset


class FullMetricDataset(Dataset):
    def __init__(self, data_csv, labels_csv, xval_idx=0, is_train=True,
                 ablation_feat=None, drop_feats=["Conversation ID", "Mean inter-speaker silence", "Interruption rate"]):
        self.is_train = is_train

        if ablation_feat is not None:
            drop_feats.append(ablation_feat)

        data_df = pd.read_csv(data_csv, delimiter='\t').drop(drop_feats, axis=1)
        self.data_np = data_df.to_numpy()
        self.data_scaler = StandardScaler()

        labels_df = pd.read_csv(labels_csv).drop(["treatment", "conversation"], axis=1)
        self.labels_np = labels_df.to_numpy()
        self.labels_scaler = StandardScaler()

        if xval_idx > 9:
            raise Exception("Invalid cross validation index for Keeper dataset")

        if self.is_train:
            if xval_idx != 9:
                self.data_np = np.concatenate([self.data_np[:2*xval_idx],
                                               self.data_np[2*(xval_idx+1):]], axis=0)
                self.labels_np = np.concatenate([self.labels_np[:2*xval_idx],
                                                 self.labels_np[2*(xval_idx+1):]], axis=0)
            elif xval_idx == 9:
                self.data_np = self.data_np[:2*xval_idx]
                self.labels_np = self.labels_np[:2*xval_idx]

            self.data_scaled = self.data_scaler.fit_transform(self.data_np)
            self.labels_scaled = self.labels_scaler.fit_transform(self.labels_np)
        else:
            if xval_idx != 9:
                self.data_np = self.data_np[2*xval_idx:2*(xval_idx+1)]
                self.labels_np = self.labels_np[2*xval_idx:2*(xval_idx+1)]
            elif xval_idx == 9:
                self.data_np = self.data_np[2*xval_idx:]
                self.labels_np = self.labels_np[2*xval_idx:]

    def __len__(self):
        return len(self.data_np)

    def __getitem__(self, i):
        if self.is_train:
            data, labels = torch.tensor(self.data_scaled[i], dtype=torch.float), torch.tensor(self.labels_scaled[i], dtype=torch.float)
        else:
            data, labels = torch.tensor(self.data_np[i], dtype=torch.float), torch.tensor(self.labels_np[i], dtype=torch.float)
        return data, labels


class SingleMetricDataset(Dataset):
    def __init__(self, data_csv, labels_csv, pred_val, xval_idx=0,
                 is_train=True, ablation_feat=None,
                 drop_feats=["Conversation ID", "Mean inter-speaker silence", "Interruption rate"]):
        self.is_train = is_train

        if ablation_feat is not None:
            drop_feats.append(ablation_feat)

        data_df = pd.read_csv(data_csv, delimiter='\t').drop(drop_feats, axis=1)
        self.data_np = data_df.to_numpy()
        self.data_scaler = StandardScaler()

        labels_df = pd.read_csv(labels_csv)[pred_val]
        self.labels_np = labels_df.to_numpy()
        self.labels_scaler = StandardScaler()

        if xval_idx > 9:
            raise Exception("Invalid cross validation index for Keeper dataset")

        if self.is_train:
            if xval_idx != 9:
                self.data_np = np.concatenate([self.data_np[:2*xval_idx],
                                               self.data_np[2*(xval_idx+1):]], axis=0)
                self.labels_np = np.concatenate([self.labels_np[:2*xval_idx],
                                                 self.labels_np[2*(xval_idx+1):]], axis=0)
            elif xval_idx == 9:
                self.data_np = self.data_np[:2*xval_idx]
                self.labels_np = self.labels_np[:2*xval_idx]

            self.data_scaled = self.data_scaler.fit_transform(self.data_np)
            self.labels_scaled = self.labels_scaler.fit_transform(self.labels_np)
        else:
            if xval_idx != 9:
                self.data_np = self.data_np[2*xval_idx:2*(xval_idx+1)]
                self.labels_np = self.labels_np[2*xval_idx:2*(xval_idx+1)]
            elif xval_idx == 9:
                self.data_np = self.data_np[2*xval_idx:]
                self.labels_np = self.labels_np[2*xval_idx:]

    def __len__(self):
        return len(self.data_np)

    def __getitem__(self, i):
        if self.is_train:
            data, labels = torch.tensor(self.data_scaled[i], dtype=torch.float), torch.tensor(self.labels_scaled[i], dtype=torch.float)
        else:
            data, labels = torch.tensor(self.data_np[i], dtype=torch.float), torch.tensor(self.labels_np[i], dtype=torch.float)
        return data, labels


