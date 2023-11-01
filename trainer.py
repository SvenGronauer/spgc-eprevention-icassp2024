import torch
import torch.nn as nn
import sklearn.metrics
import os
import numpy as np

class Trainer:

    ''' Class to train the classifier '''

    def __init__(self, models, optims, scheds, loaders, args):

        self.models = models
        self.optims = optims
        self.scheds = scheds
        self.dataloaders = loaders
        self.args = args
        self.criterion = nn.MSELoss()

        self.current_best_avgs = [0 for _ in range(len(self.models))]
        self.current_best_aurocs = [0 for _ in range(len(self.models))]
        self.current_best_auprcs = [0 for _ in range(len(self.models))]


    def train(self):
        # Initialize output metrics

        # Process each epoch
        for epoch in range(self.args.epochs):
            for i in range(len(self.models)):

                # ------ train one epoch ------ #

                epoch_metrics = {}

                self.models[i].train()
                torch.set_grad_enabled(True)

                for batch in self.dataloaders[i]['train']:

                    if batch is None:
                        continue

                    x = batch['data'].to(self.args.device)

                    # Forward
                    logits, features = self.models[i](x[:, :-1])

                    # Compute loss and backprop
                    loss = self.criterion(logits, x[:, -1:].permute(0, 2, 1))

                    self.optims[i].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.models[i].parameters(), 5)

                    self.optims[i].step()


                    # Log metrics
                    metrics = {
                            'loss': loss.item(),
                        }

                    for k, v in metrics.items():
                        epoch_metrics[k] = epoch_metrics[k] + [v] if k in epoch_metrics else [v]

                self.scheds[i].step()


                # ------ start validating ------ #
                self.models[i].eval()
                torch.set_grad_enabled(False)

                # ---------- run on train distribution loader to get the losses in eval mode ---------- #

                all_labels_train = []
                all_losses_train = []

                for batch in self.dataloaders[i]['train_distribution']:
                    if batch is None:
                         continue

                    x = batch['data'].to(self.args.device)
                    user_ids = batch['user_id'].to(self.args.device)
                    logits, features = self.models[i](x.squeeze(0)[:, :-1])
                    loss = self.criterion(logits, x.squeeze(0)[:, -1:].permute(0, 2, 1))
                    all_losses_train.append(loss.item())
                    all_labels_train.append(user_ids.detach().cpu())

                all_losses_train = torch.tensor(all_losses_train)
                all_labels_train = torch.hstack(all_labels_train).numpy()

                train_loss = {}

                for subject in np.unique(all_labels_train):
                    subject_losses_train = all_losses_train[all_labels_train == subject]
                    train_loss[subject] = subject_losses_train

                # ---------- run on validation loader ---------- #
                anomaly_scores = []
                relapse_labels = []
                user_ids = []

                for batch in self.dataloaders[i]['val']:

                    if batch is None:
                        continue

                    x = batch['data'].to(self.args.device)
                    user_id = batch['user_id'].to(self.args.device)

                    # Forward
                    logits, features = self.models[i](x[:, :, :-1].squeeze(0))
                    loss = self.criterion(logits, x.squeeze(0)[:, -1:].permute(0, 2, 1))

                    anomaly_score = (loss.mean().item() - train_loss[user_id.item()].mean()) / (
                                train_loss[user_id.item()].max() - train_loss[
                            user_id.item()].min())
                    anomaly_scores.append(torch.clip(anomaly_score, 0, 1).item())
                    relapse_labels.append(batch['relapse_label'].item())
                    user_ids.append(batch['user_id'].item())

                anomaly_scores = (np.array(anomaly_scores) > 0.0).astype(np.float64)
                relapse_labels = np.array(relapse_labels)
                user_ids = np.array(user_ids)

                # Calculating metrics

                all_auroc = []
                all_auprc = []

                # calculate for each user separately
                for user in np.unique(user_ids):
                    user_anomaly_scores = anomaly_scores[user_ids==user]
                    user_relapse_labels = relapse_labels[user_ids==user]

                    # Compute ROC Curve
                    precision, recall, _ = sklearn.metrics.precision_recall_curve(user_relapse_labels, user_anomaly_scores)

                    fpr, tpr, _ = sklearn.metrics.roc_curve(user_relapse_labels, user_anomaly_scores)

                    # # Compute AUROC
                    auroc = sklearn.metrics.auc(fpr, tpr)

                    # # Compute AUPRC
                    auprc = sklearn.metrics.auc(recall, precision)

                    all_auroc.append(auroc)
                    all_auprc.append(auprc)

                total_auroc = sum(all_auroc)/len(all_auroc)
                total_auprc = sum(all_auprc)/len(all_auprc)
                total_avg = (total_auroc + total_auprc) / 2


                # save best model
                if total_avg > self.current_best_avgs[i]:
                    self.current_best_avgs[i] = total_avg
                    self.current_best_auprcs[i] = total_auprc
                    self.current_best_aurocs[i] = total_auroc
                    os.makedirs(self.args.save_path, exist_ok=True)
                    if not os.path.exists(os.path.join(self.args.save_path, str(i))):
                        os.mkdir(f'{self.args.save_path}/{i}')
                    torch.save(self.models[i].state_dict(), os.path.join(self.args.save_path, f'{i}/best_model.pth'))

            print("Epoch ", epoch)
            for i in range(len(self.models)):
                print("P"+str(i+1), f'Total AUROC: {self.current_best_aurocs[i]:.4f}, Total AUPRC: {self.current_best_auprcs[i]:.4f}, Total AVG: {self.current_best_avgs[i]:.4f}')

        

