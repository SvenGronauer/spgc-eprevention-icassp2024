import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import sklearn.metrics
import os
from model import EnsembleLinear


class Trainer:
    ''' Class to train the classifier '''

    def __init__(self, model, optim, sched, loaders, args):

        self.model = model
        self.optim = optim
        self.sched = sched
        self.dataloaders = loaders
        self.args = args
        self.criterion = nn.MSELoss()

        self.current_best_score = 0

        self.mlps = []
        self.optimizers = []
        for i in range(self.args.num_patients):

            # FIXME: change Linear to MLP
            ensemble_head = EnsembleLinear(
                in_features=self.args.d_model,
                out_features=self.args.output_dim,
                ensemble_size=self.args.ensembles
            )
            ensemble_head.to(self.args.device)

            self.mlps.append(ensemble_head)
            ps = ensemble_head.parameters()
            self.optimizers.append(torch.optim.Adam(ps, lr=1e-3, weight_decay=1e-4))

    def train_encoder_once(self, epoch: int):
        epoch_metrics = {}
        for batch in tqdm(self.dataloaders['train'], desc=f'Train Encoder ({epoch}/{self.args.epochs})'):

            if batch is None:
                continue

            # x has shape: (16, 6, 24) == (batch_size, input_features, seq_len)
            x = batch['data'].to(self.args.device)
            labels = batch['target'].to(self.args.device)  # shape: (batch_size, input_features, 1)
            # labels.shape : (16, 6, 24)
            labels = torch.squeeze(labels, dim=-1)

            # Forward
            features, mean = self.model(x)

            mse_loss = self.criterion(mean, labels)

            self.optim.zero_grad()
            mse_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

            self.optim.step()

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

    def train_ensembles(self):
        k = self.args.ensembles
        for i in tqdm(range(self.args.num_patients), desc=f"Train Ensembles"):
            ensemble_head = self.mlps[i]
            ensemble_head.train()
            optim = self.optimizers[i]
            loader_str = "P" + str(i+1) +'_train'
            for batch in self.dataloaders[loader_str]:
                if batch is None:
                    continue

                # x has shape: (16, 6, 24) == (batch_size, input_features, seq_len)
                x = batch['data'].to(self.args.device)
                targets = batch['target'].to(self.args.device)
                targets = torch.squeeze(targets, dim=-1)
                # Forward
                features, _ = self.model(x)

                batched_features = features[None, :, :].repeat([k, 1, 1])
                batched_targets = targets[None, :, :].repeat([k, 1, 1])
                # print(f"batched_features.shape: \n{batched_features.shape}")
                ensemble_mask = batch['idx'] % k

                resampled_x, r_targets = self.resample_batch(
                    batch['idx'],
                    self.dataloaders[loader_str].dataset
                )
                r_targets = torch.squeeze(r_targets, dim=-1)
                # resampled_x.shape = torch.Size([16, 6, 24])
                r_features, _ = self.model(resampled_x)
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

    def _evaluate_ensembles(self, epoch: int):
        k = self.args.ensembles
        anomaly_scores = []
        relapse_labels = []
        user_ids = []

        for batch in tqdm(self.dataloaders['val'], desc=f'Val, {epoch}/{self.args.epochs}'):

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


            features, _ = self.model(x)

            batched_features = features[None, :, :].repeat([k, 1, 1])
            # mean.shape = (ens_size, num_day_samples, output_dim)
            mean = ensemble_head.forward(batched_features)

            distances = torch.sum(torch.pow(mean - targets, 2), dim=(2, ))
            mean_dist = torch.mean(distances, 0)
            var_scores = (distances - mean_dist)**2
            anomaly_score = torch.mean(var_scores).item()

            anomaly_scores.append(anomaly_score)
            relapse_labels.append(batch['relapse_label'].item())
            user_ids.append(batch['user_id'].item())

        anomaly_scores = np.array(anomaly_scores)
        relapse_labels = np.array(relapse_labels)
        user_ids = np.array(user_ids)
        return anomaly_scores, relapse_labels, user_ids

    def calculate_metrics(self,  anomaly_scores, relapse_labels, user_ids, epoch_metrics):

        all_auroc = []
        all_auprc = []

        # calculate for each user separately
        for user in range(self.args.num_patients):
            user_anomaly_scores = anomaly_scores[user_ids == user]
            user_relapse_labels = relapse_labels[user_ids == user]

            # Compute ROC Curve
            precision, recall, _ = sklearn.metrics.precision_recall_curve(user_relapse_labels,
                                                                          user_anomaly_scores)

            fpr, tpr, _ = sklearn.metrics.roc_curve(user_relapse_labels, user_anomaly_scores)

            # # Compute AUROC
            auroc = sklearn.metrics.auc(fpr, tpr)

            # # Compute AUPRC
            auprc = sklearn.metrics.auc(recall, precision)

            all_auroc.append(auroc)
            all_auprc.append(auprc)
            print(f'USER: {user}, AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}')

        total_auroc = sum(all_auroc) / len(all_auroc)
        total_auprc = sum(all_auprc) / len(all_auprc)
        total_avg = (total_auroc + total_auprc) / 2
        print(
            f'Total AUROC: {total_auroc:.4f}, Total AUPRC: {total_auprc:.4f}, Total AVG: {total_avg:.4f}, Train Loss: {np.mean(epoch_metrics["loss"]):.4f}')

        # save best model
        if total_avg > self.current_best_score:
            self.current_best_score = total_avg
            os.makedirs(self.args.save_path, exist_ok=True)
            torch.save(self.model.state_dict(),
                       os.path.join(self.args.save_path, f'best_model.pth'))
            print('Saved best model!')

    def train(self):
        # Initialize output metrics

        # Process each epoch
        for epoch in range(self.args.epochs):
            print("*"*55)
            print(f"Start epoch {epoch}/{self.args.epochs}")

            self.model.train()
            torch.set_grad_enabled(True)

            epoch_metrics = self.train_encoder_once(epoch)

            self.sched.step()

            self.model.eval()
            self.train_ensembles()

            torch.set_grad_enabled(False)

            print('Calculating accuracy on validation set and anomaly scores...')
            anomaly_scores, relapse_labels, user_ids = self._evaluate_ensembles(epoch)

            print('Calculating metrics...')
            self.calculate_metrics(anomaly_scores, relapse_labels, user_ids, epoch_metrics)

        
