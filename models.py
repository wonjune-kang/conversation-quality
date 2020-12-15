import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleRegressionLinear(nn.Module):
    def __init__(self, n_feats):
        super(SingleRegressionLinear, self).__init__()
        self.fc_out = nn.Linear(n_feats, 1)

    def forward(self, x):
        x = self.fc_out(x)
        return x

class MultipleRegressionLinear(nn.Module):
    def __init__(self, n_feats, n_out):
        super(SingleRegressionLinear, self).__init__()
        self.fc_out = nn.Linear(n_feats, n_out)

    def forward(self, x):
        x = self.fc_out(x)
        return x

class SingleRegressionNet(nn.Module):
    def __init__(self, n_feats, n_hidden):
        super(SingleRegressionNet, self).__init__()
        self.fc_1 = nn.Linear(n_feats, n_hidden)
        self.fc_2 = nn.Linear(n_hidden, n_hidden)
        self.fc_out = nn.Linear(n_hidden, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc_1(x))
        x = self.dropout(x)
        x = self.relu(self.fc_2(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        return x

class MultipleRegressionNet(nn.Module):
    def __init__(self, n_feats, n_hidden, n_out=4):
        super(MultipleRegressionNet, self).__init__()
        self.fc_1 = nn.Linear(n_feats, n_hidden)
        self.fc_2 = nn.Linear(n_hidden, n_hidden)
        self.fc_out = nn.Linear(n_hidden, n_out)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc_1(x))
        x = self.dropout(x)
        x = self.relu(self.fc_2(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        return x



