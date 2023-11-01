import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import sklearn.metrics
# from sklearn.svm import OneClassSVM
import os
import numpy as np
from sklearn.covariance import EllipticEnvelope

class Trainer:

    ''' Class to train the classifier '''

    def __init__(self, model, optim, sched, loaders, args):

        self.model = model
        self.optim = optim
        self.sched = sched
        self.dataloaders = loaders
        self.args = args
        self.criterion = nn.CrossEntropyLoss()


        self.current_best_score = 0


    def train(self):
        # Initialize output metrics

        # Process each epoch
        for epoch in range(self.args.epochs):

            # ------ train one epoch ------ #

            epoch_metrics = {}

            self.model.train()
            torch.set_grad_enabled(True)

            for batch in tqdm(self.dataloaders['train'], desc=f'Train, {epoch}/{self.args.epochs}'):
                
                if batch is None:
                    continue

                # x has shape: (16, 8, 23) == (batch_size, input_features, seq_len-1)
                x = batch['data'].to(self.args.device)
                user_ids = batch['user_id'].to(self.args.device)
                labels = batch['target'].to(self.args.device)  # of shape: (batch_size, input_features, 1)
                labels = torch.squeeze(labels, dim=-1)
                labels = labels[None, :, :].repeat([self.args.ensembles, 1, 1])  # to shape: (ens_size, batch_size, input_features)

                # Forward
                logits, features, mean = self.model(x)

                # prediction loss
                mse_loss = torch.sum(torch.pow(mean - labels, 2), dim=(0, 2))
                total_loss = torch.mean(mse_loss)

                # calculate accuracy
                output_probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(output_probabilities, dim=1)

                acc = (predicted_class == user_ids).float().mean()

                self.optim.zero_grad()

                # Compute loss and backprop
                # loss = self.criterion(logits, user_ids)
                # loss.backward()

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

                self.optim.step()


                # Log metrics
                metrics = {
                        'loss': total_loss.item(),
                        'acc': acc.item(),
                    }

                for k, v in metrics.items():
                    epoch_metrics[k] = epoch_metrics[k] + [v] if k in epoch_metrics else [v]

            self.sched.step()


            # ------ start validating ------ #
            self.model.eval()
            torch.set_grad_enabled(False)

            # ---------- run on validation loader ---------- #
            print('Calculating accuracy on validation set and anomaly scores...')
            anomaly_scores = []
            relapse_labels = []
            user_ids = []

            for batch in tqdm(self.dataloaders['val'], desc=f'Val, {epoch}/{self.args.epochs}'):

                if batch is None:
                    continue
                # data is of size: (1, windowsPerDay, 8, 23)
                x = batch['data'].to(self.args.device)
                user_id = batch['user_id'].to(self.args.device)
                # labels are of size: (1, 16, 8, 1)
                labels = batch['target'].to(self.args.device)
                labels = torch.squeeze(labels, 3)
                labels = torch.squeeze(labels, 0)
                labels = labels[None, :, :].repeat([self.args.ensembles, 1, 1])  # to shape: (ens_size, batch_size, input_features)

                x = x.squeeze(0) # remove fake batch dimension

                # Forward
                logits, features, mean = self.model(x)

                # Notes:
                # - calculate mean anomaly score for whole day
                # - anomaly score is Mahalanobis distance

                scores = torch.sum(torch.pow(mean - labels, 2), dim=(0, 2)) - 100
                anomaly_score = torch.mean(scores).item()
                anomaly_scores.append(anomaly_score)
                relapse_labels.append(batch['relapse_label'].item())
                user_ids.append(batch['user_id'].item())
            
            anomaly_scores = np.array(anomaly_scores)
            relapse_labels = np.array(relapse_labels)
            user_ids = np.array(user_ids)

            print('Calculating metrics...')

            all_auroc = []
            all_auprc = []

            # calculate for each user separately
            for user in range(self.args.num_patients):
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
                print(f'USER: {user}, AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}')

            total_auroc = sum(all_auroc)/len(all_auroc)
            total_auprc = sum(all_auprc)/len(all_auprc)
            total_avg = (total_auroc + total_auprc) / 2
            print(f'Total AUROC: {total_auroc:.4f}, Total AUPRC: {total_auprc:.4f}, Total AVG: {total_avg:.4f}, Train Loss: {np.mean(epoch_metrics["loss"]):.4f}, Train Acc: {np.mean(epoch_metrics["acc"]):.4f}')


            # save best model
            if total_avg > self.current_best_score:
                self.current_best_score = total_avg
                os.makedirs(self.args.save_path, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(self.args.save_path, f'best_model.pth'))
                print('Saved best model!')

        

