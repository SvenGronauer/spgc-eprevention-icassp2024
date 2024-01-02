import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import sklearn.metrics
import os
from model import EnsembleLinear


def create_ensemble_mlp(args):
    m = nn.Sequential(
                EnsembleLinear(args.d_model,args.d_model,args.ensembles),
                nn.ReLU(),
                EnsembleLinear(args.d_model,args.d_model,args.ensembles),
                nn.ReLU(),
                EnsembleLinear(args.d_model,args.output_dim,args.ensembles),
    )
    m.to(args.device)
    return m

class Trainer:
    ''' Class to train the classifier '''

    def __init__(self, models, optims, scheds, loaders, args):

        self.models = models
        self.optims = optims
        self.scheds = scheds
        self.dataloaders = loaders
        self.args = args
        self.criterion = nn.MSELoss()

        self.current_best_avgs = [-np.inf for _ in range(len(self.models))]
        self.current_best_aurocs = [-np.inf for _ in range(len(self.models))]
        self.current_best_auprcs = [-np.inf for _ in range(len(self.models))]

        self.mlps = []
        self.optimizers = []
        for i in range(len(self.models)):

            ensemble_head = create_ensemble_mlp(self.args)

            self.mlps.append(ensemble_head)
            ps = ensemble_head.parameters()
            self.optimizers.append(torch.optim.Adam(ps, lr=1e-3, weight_decay=1e-4))

    def train_encoder_once(self, epoch: int, i: int, epoch_metrics: dict):
        for batch in tqdm(self.dataloaders[i]['train'], desc=f'Train Encoder ({i+1}/{len(self.models)})'):

            if batch is None:
                continue

            # x has shape: (16, 6, 24) == (batch_size, input_features, seq_len)
            x = batch['data'].to(self.args.device)
            labels = batch['target'].to(self.args.device)  # shape: (batch_size, input_features, 1)
            # labels.shape : (16, 6, 24)
            labels = torch.squeeze(labels, dim=-1)

            # Forward
            features, mean = self.models[i](x)

            mse_loss = self.criterion(mean, labels)

            self.optims[i].zero_grad()
            mse_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.models[i].parameters(), 5)

            self.optims[i].step()

            # Log metrics
            metrics = {
                'loss': mse_loss.item(),
            }

            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics[k] + [v] if k in epoch_metrics else [v]

        return epoch_metrics

    def resample_batch(self, indices, dataset):
        ensemble_size = self.args.ensembles
        offsets = torch.zeros_like(indices)
        data_batch = []
        target_batch = []
        for i in range(indices.size()[0]):
            while (offsets[i] % ensemble_size) == 0:
                offsets[i] = torch.randint(low=1, high=len(dataset), size=(1,))
            r_idx = (offsets[i] + indices[i]) % len(dataset)
            x = dataset[r_idx]['data']
            t = dataset[r_idx]['target']
            # print(x.shape)
            data_batch.append(x)
            target_batch.append(t)
        data = torch.stack(data_batch).to(self.args.device)
        targets = torch.stack(target_batch).to(self.args.device)
        return data, targets

    def train_ensemble(self, i: int):
        k = self.args.ensembles
        ensemble_head = self.mlps[i]
        ensemble_head.train()
        optim = self.optimizers[i]
        size = len(self.dataloaders[i]['train'])

        for batch in tqdm(self.dataloaders[i]['train'], desc=f'Train Ensemble ({i+1}/{len(self.models)})'):
            if batch is None:
                continue

            # x has shape: (16, 6, 24) == (batch_size, input_features, seq_len)
            x = batch['data'].to(self.args.device)
            targets = batch['target'].to(self.args.device)
            targets = torch.squeeze(targets, dim=-1)
            # Forward
            features, _ = self.models[i](x)

            batched_features = features[None, :, :].repeat([k, 1, 1])
            batched_targets = targets[None, :, :].repeat([k, 1, 1])
            # print(f"batched_features.shape: \n{batched_features.shape}")
            ensemble_mask = batch['idx'] % k

            resampled_x, r_targets = self.resample_batch(
                batch['idx'],
                self.dataloaders[i]['train'].dataset
            )
            r_targets = torch.squeeze(r_targets, dim=-1)
            # resampled_x.shape = torch.Size([16, 6, 24])
            r_features, _ = self.models[i](resampled_x)
            # print(f"r_features.shape: \n{r_features.shape}")
            batched_features[ensemble_mask] = r_features
            batched_targets[ensemble_mask] = r_targets

            # forward pass
            mean = ensemble_head.forward(batched_features)
            mse_loss = torch.sum(torch.pow(mean - batched_targets, 2), dim=(0, 2))
            total_loss = torch.mean(mse_loss)

            optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(ensemble_head.parameters(), 5)

            optim.step()

    def evaluate_ensembles(self, epoch: int, i: int, train_dist_anomaly_scores: dict):
        k = self.args.ensembles
        anomaly_scores = []
        relapse_labels = []
        user_ids = []

        _mean = np.mean(train_dist_anomaly_scores[i])
        _max, _min = np.max(train_dist_anomaly_scores[i]), np.min(train_dist_anomaly_scores[i])

        for batch in self.dataloaders[i]['val']:

            if batch is None:
                continue
            user_id = batch['user_id'].to(self.args.device)
            if user_id.item() >= self.args.num_patients:
                continue

            # .Note: using batch_size = 1  -- x.shape = (1, 16, 6, 24)
            x = batch['data'].to(self.args.device)
            x = x.squeeze(0)  # remove fake batch dimension
            ensemble_head = self.mlps[user_id.item()]
            ensemble_head.eval()
            # targets.shape = (1, 16, 2, 24)
            targets = batch['target'].to(self.args.device)
            targets = torch.squeeze(targets)
            if targets.ndim < 2:
                targets = torch.reshape(targets, (1, -1))

            # targets.shape: (ens_size, num_day_samples, out_dim)
            targets = targets[None, :, :].repeat([k, 1, 1])

            features, _ = self.models[i](x)

            batched_features = features[None, :, :].repeat([k, 1, 1])
            # mean.shape = (ens_size, num_day_samples, output_dim)
            mean = ensemble_head.forward(batched_features)

            distances = torch.sum(torch.pow(mean - targets, 2), dim=(2, ))
            mean_dist = torch.mean(distances, 0)
            var_scores = (distances - mean_dist)**2
            mean_var = torch.mean(var_scores).item()
            anomaly_score = (mean_var - _mean) / (_max - _min)

            anomaly_scores.append(anomaly_score)
            relapse_labels.append(batch['relapse_label'].item())
            user_ids.append(batch['user_id'].item())

        anomaly_scores = (np.array(anomaly_scores) > 0.0).astype(np.float64)
        relapse_labels = np.array(relapse_labels)
        user_ids = np.array(user_ids)
        return anomaly_scores, relapse_labels, user_ids

    def calculate_metrics(self, user: int, anomaly_scores, relapse_labels, user_ids, epoch_metrics):

        assert np.unique(user_ids) == user

        # Compute ROC Curve
        precision, recall, _ = sklearn.metrics.precision_recall_curve(relapse_labels,anomaly_scores)

        fpr, tpr, _ = sklearn.metrics.roc_curve(relapse_labels, anomaly_scores)

        # # Compute AUROC
        auroc = sklearn.metrics.auc(fpr, tpr)

        # # Compute AUPRC
        auprc = sklearn.metrics.auc(recall, precision)

        avg = (auroc + auprc) / 2
        print(f'\tUSER: {user}, AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, AVG: {avg:.4f}')
        return auroc, auprc

    def get_train_dist_anomaly_scores(self, epoch: int, i: int):
        k = self.args.ensembles
        anomaly_scores = []
        for batch in self.dataloaders[i]['train_dist']:

            if batch is None:
                continue

            x = batch['data'].to(self.args.device)
            ensemble_head = self.mlps[i]
            targets = batch['target'].to(self.args.device)
            targets = torch.squeeze(targets, dim=-1)
            batched_targets = targets[None, :, :].repeat([k, 1, 1])

            features, _ = self.models[i](x)

            batched_features = features[None, :, :].repeat([k, 1, 1])
            # mean.shape = (ens_size, num_day_samples, output_dim)
            mean = ensemble_head.forward(batched_features)

            distances = torch.sum(torch.pow(mean - batched_targets, 2), dim=(2,))
            mean_dist = torch.mean(distances, 0)
            var_scores = (distances - mean_dist) ** 2
            anomaly_score = torch.mean(var_scores).item()
            anomaly_scores.append(anomaly_score)
        return anomaly_scores

    def validate(self, epoch: int, epoch_metrics: dict, train_dist_anomaly_scores: dict):
        for i in range(len(self.models)):
            # print('Calculating accuracy on validation set and anomaly scores...')
            anomaly_scores, relapse_labels, user_ids = self.evaluate_ensembles(epoch, i, train_dist_anomaly_scores)

            # print('Calculating metrics...')
            auroc, auprc = self.calculate_metrics(i, anomaly_scores, relapse_labels, user_ids, epoch_metrics)

            # save best model
            avg = (auroc + auprc) / 2
            if avg > self.current_best_avgs[i]:
                self.current_best_avgs[i] = max(avg, self.current_best_avgs[i])
                self.current_best_aurocs[i] = max(auroc, self.current_best_aurocs[i])
                self.current_best_auprcs[i] = max(auprc, self.current_best_auprcs[i])
                os.makedirs(self.args.save_path, exist_ok=True)
                if not os.path.exists(os.path.join(self.args.save_path, str(i))):
                    os.mkdir(f'{self.args.save_path}/{i}')
                torch.save(self.models[i].state_dict(),
                           os.path.join(self.args.save_path, f'{i+1}/best_encoder.pth'))
                torch.save(self.mlps[i].state_dict(),
                           os.path.join(self.args.save_path, f'{i+1}/best_ensembles.pth'))

        for i in range(len(self.models)):
            print(f"P{str(i + 1)} AUROC: {self.current_best_aurocs[i]:.4f}, "
                  f"AUPRC: {self.current_best_auprcs[i]:.4f}, "
                  f"AVG: {self.current_best_avgs[i]:.4f}")

        total_auroc = sum(self.current_best_aurocs) / len(self.models)
        total_auprc = sum(self.current_best_auprcs) / len(self.models)
        total_avg = (total_auroc + total_auprc) / 2

        print(f'TOTAL\tAUROC: {total_auroc:.4f},  AUPRC: {total_auprc:.4f}, Total AVG: {total_avg:.4f}, '
              f'Train Loss: {np.mean(epoch_metrics["loss"]):.4f}')


    def train(self):
        for epoch in range(self.args.epochs):
            print("*"*55)
            print(f"Start epoch {epoch+1}/{self.args.epochs}")
            epoch_metrics = {}
            train_dist_anomaly_scores = {}
            for i in range(len(self.models)):

                # ------ start training ------ #
                self.models[i].train()
                epoch_metrics = self.train_encoder_once(epoch, i, epoch_metrics)
                self.scheds[i].step()

                self.models[i].eval()
                self.train_ensemble(i)
                with torch.no_grad():
                    train_dist_anomaly_scores[i] = self.get_train_dist_anomaly_scores(epoch, i)

            # ------ start validating ------ #
            with torch.no_grad():
                self.validate(epoch, epoch_metrics, train_dist_anomaly_scores)
