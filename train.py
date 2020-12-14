import os
import csv
import argparse
import numpy as np
import torch
from tqdm import tqdm

from data_load import *
from models import *


def compute_mse_per_feat(y, y_pred, data_scaler):
    y_unscaled = data_scaler.inverse_transform(y)
    y_pred_unscaled = data_scaler.inverse_transform(y_pred)
    unscaled_mse = (np.square(y_unscaled - y_pred_unscaled)).mean(axis=0)
    return unscaled_mse

def get_single_metric_dataloaders(data_csv, labels_csv, pred_category, xval_idx=0, ablation_feat=None):
    train_dataset = SingleMetricDataset(data_csv, labels_csv, pred_category,
                                        xval_idx=xval_idx, is_train=True, ablation_feat=ablation_feat)
    val_dataset = SingleMetricDataset(data_csv, labels_csv, pred_category,
                                      xval_idx=xval_idx, is_train=False, ablation_feat=ablation_feat)

    if xval_idx == 9:
        train_batch_size = 18
        val_batch_size = 3
    else:
        train_batch_size = 19
        val_batch_size = 2

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True,
                                               num_workers=4,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=val_batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            drop_last=True)

    return train_loader, val_loader, train_dataset, val_dataset, train_batch_size, val_batch_size

def get_full_metric_dataloaders(data_csv, labels_csv, xval_idx=0, ablation_feat=None):
    train_dataset = FullMetricDataset(data_csv, labels_csv, xval_idx=xval_idx,
                                      is_train=True, ablation_feat=ablation_feat)
    val_dataset = FullMetricDataset(data_csv, labels_csv, xval_idx=xval_idx,
                                    is_train=False, ablation_feat=ablation_feat)

    if xval_idx == 9:
        train_batch_size = 18
        val_batch_size = 3
    else:
        train_batch_size = 19
        val_batch_size = 2

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True,
                                               num_workers=1,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=val_batch_size,
                                            shuffle=True,
                                            num_workers=1,
                                            drop_last=True)

    return train_loader, val_loader, train_dataset, val_dataset, train_batch_size, val_batch_size


def train(model, dataloader_info, device, iterations, criterion, optimizer,
          scheduler, ckpt_path, xval_idx, run_idx):

    train_loader, val_loader, train_dataset, val_dataset, train_batch_size, val_batch_size = dataloader_info

    best_iter = -1
    best_val_loss = np.inf
    best_avg_train_loss = np.inf
    best_avg_train_mse = np.inf
    best_avg_val_loss = np.inf
    best_avg_val_mse = np.inf

    print(f"\nCross validation fold {xval_idx}, run {run_idx}:")
    for iteration in tqdm(range(1, iterations+1)):
        model.train()

        train_loss = 0.0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)
            loss = criterion(y_pred, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mse = compute_mse_per_feat(y.cpu().detach().numpy(), y_pred.cpu().detach().numpy(),
                                             train_dataset.labels_scaler)

        with torch.no_grad():
            val_loss = 0.0
            for X_val, y_val in val_loader:
                X_val = torch.tensor(train_dataset.data_scaler.transform(X_val.cpu().detach().numpy())).float()
                y_val = torch.tensor(train_dataset.labels_scaler.transform(y_val.cpu().detach().numpy())).float()

                X_val = X_val.to(device)
                y_val = y_val.to(device)

                y_pred_category = model(X_val)
                loss = criterion(y_pred_category, y_val)

                val_loss += loss.item()
                val_mse = compute_mse_per_feat(y_val.cpu().detach().numpy(), y_pred_category.cpu().detach().numpy(),
                                               train_dataset.labels_scaler)

        scheduler.step()

        avg_train_loss = train_loss/train_batch_size
        avg_val_loss = val_loss/val_batch_size

        if avg_val_loss < best_val_loss:
            best_iter = iteration
            best_val_loss = avg_val_loss
            best_avg_train_loss = avg_train_loss
            best_avg_train_mse = train_mse
            best_avg_val_loss = avg_val_loss
            best_avg_val_mse = val_mse

            model.eval().cpu()
            ckpt_filename = f"best_fold_{xval_idx}_run_{run_idx}.model"
            ckpt_save_path = os.path.join(ckpt_path, ckpt_filename)
            torch.save(model.state_dict(), ckpt_save_path)
            model.to(device).train()
    
    print("\nAverage training loss: {0:.3f}".format(best_avg_train_loss))
    print("Average training MSE:", best_avg_train_mse.round(3))
    print("\nAverage validation loss: {0:.3f}".format(best_avg_val_loss))
    print("Average validation MSE:", best_avg_val_mse.round(3))
    print(f"\nSaved model checkpoint for best iteration {best_iter} to {ckpt_save_path}")

    return best_avg_val_mse


FEAT_NAMES = ["Words per hour", "Speech per turn", "Turn taking balance",
              "Grade level", "MATTR lexical diversity", "Mean word length",
              "VADER sentiment", "Responsivity rate"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "MultipleRegressionNet")

    parser.add_argument('--data_csv',        type=str,   default='./data/csv/keeper_metrics_full.csv', help='Path to CSV file with pre-extracted features')
    parser.add_argument('--labels_csv',      type=str,   default='./data/csv/keeper_survey_avg.csv',   help='Path to CSV file with labels')

    parser.add_argument('--mult_regression', type=bool,  default=True,   help='Whether to predict all label categories simultaneously or not')
    parser.add_argument('--pred_category',   type=str,                   help='Label category to predict if single regression')
    parser.add_argument('--ablation',        type=bool,  default=False,  help='Whether to ablate a feature')
    parser.add_argument('--ablation_feat',   type=str,                   help='Input feature to be ablated'), 

    parser.add_argument('--iterations',      type=int,   default=2000,   help='Number of training iterations')
    parser.add_argument('--lr',              type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--lr_step',         type=int,   default=400,    help='Learning rate scheduler step')
    parser.add_argument('--n_hidden',        type=int,   default=10,     help='Number of hidden units for network')
    parser.add_argument('--runs_per_fold',   type=int,   default=5,      help='Number of runs for each cross validation fold')

    parser.add_argument('--ckpt_path',       type=str,   default="./checkpoints",  help='Path to save model checkpoints')
    parser.add_argument('--log_file',        type=str,   default="./logs/log.csv", help='Path to log file')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda:0"):
        print("Using GPU.")
    else:
        print("Using CPU.")

    os.makedirs(args.ckpt_path, exist_ok=True)

    if args.ablation == True:
        if args.ablation_feat not in FEAT_NAMES:
            raise Exception("Invalid feature name for ablation.")
        n_feats = 7
        ablation_feat = args.ablation_feat
        print(f"Ablating on feature: {ablation_feat}")
    else:
        n_feats = 8
        ablation_feat = None

    with open(args.log_file, 'w', newline='') as log_file:
        csv_writer = csv.writer(log_file, delimiter='\t')
        csv_writer.writerow(['fold', 'ease', 'sp', 'anxiety', 'tone'])

        for xval_idx in range(10):
            for run_idx in range(1, args.runs_per_fold+1):
                if args.mult_regression:
                    model = MultipleRegressionNet(n_feats, args.n_hidden)
                    dataloader_info = get_full_metric_dataloaders(args.data_csv, args.labels_csv, xval_idx, ablation_feat)
                else:
                    model = SingleRegressionNet(n_feats, args.n_hidden)
                    dataloader_info = get_single_metric_dataloaders(args.data_csv, args.labels_csv, args.pred_category, xval_idx, ablation_feat)
                model = model.to(device)

                criterion = torch.nn.MSELoss()
                optimizer = torch.optim.Adam([
                                {'params': model.parameters()},
                                {'params': criterion.parameters()}
                            ], lr=args.lr)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.5)

                avg_val_mse = train(model, dataloader_info, device, args.iterations,
                                    criterion, optimizer, scheduler, args.ckpt_path,
                                    xval_idx, run_idx)

                fold = f"fold_{xval_idx}_run_{run_idx}"
                csv_writer.writerow([fold]+[round(x, 3) for x in avg_val_mse.tolist()])

    log_file.close()

        


