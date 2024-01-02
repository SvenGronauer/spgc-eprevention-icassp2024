from pprint import pprint

import torch
import argparse
from model import TransformerClassifier
from dataset import PatientDataset
import pickle
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import sklearn
from sklearn.svm import OneClassSVM

from trainer import create_ensemble_mlp

def calculate_sincos_from_minutes(minutes):
    time_value = minutes * (2. * np.pi / (60 * 24))
    sin_t = np.sin(time_value)
    cos_t = np.cos(time_value)
    return sin_t, cos_t

def parse():
    '''Returns args passed to the train.py script.'''
    parser = argparse.ArgumentParser()

    # transformer parameters
    parser.add_argument('--window_size', type=int, default=48)
    parser.add_argument('--input_features', type=int, default=8)
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--ensembles', type=int, default=5)

    # num_patients - 9 for track 1, 8 for track 2
    parser.add_argument('--num_patients', type=int, default=9)

    # input paths
    parser.add_argument('--features_path', type=str, help='features to use')
    parser.add_argument('--dataset_path', type=str, help='dataset path') # to get relapse labels
    parser.add_argument('--submission_path', type=str, help='where to save the submission files', default='/var/tmp/spgc-submission') # to get relapse labels

    # checkpoint
    parser.add_argument('--load_path', type=str, help='path to root directory of models')

    parser.add_argument('--device', type=str, help='device to use (cpu, cuda, cuda[number])', default='cpu')

    parser.add_argument('--mode', type=str, help='val, test', default='test')

    args = parser.parse_args()
    args.seq_len = args.window_size
            
    return args

def main():

    # parse arguments
    args = parse()
    window_size = args.window_size

    pprint(args)

    print(args.d_model)

    columns_to_scale = ['acc_norm', 'gyr_norm', 'heartRate_mean', 'rRInterval_mean',
                        'rRInterval_rmssd', 'rRInterval_sdnn', 'rRInterval_lombscargle_power_high',
                        'steps']
    data_columns = columns_to_scale
    target_columns = ['sin_t', 'cos_t']

    device = args.device
    print('Using device', args.device)

    # Models
    encoders = [TransformerClassifier(vars(args)).to(device) for _ in range(args.num_patients)]
    mlps = [create_ensemble_mlp(args) for _ in range(args.num_patients)]

    with open(os.path.join(args.load_path, "train_dist_anomaly_scores.pkl"), 'rb') as f:
        train_dist_anomaly_scores = pickle.load(f)

    # dataset scalers
    scalers = []

    # load checkpoints
    for i in range(len(encoders)):
        pth = os.path.join(args.load_path, str(i+1))
        checkpoint = torch.load(os.path.join(pth, 'best_encoder.pth'), map_location=torch.device('cpu'))
        encoders[i].load_state_dict(checkpoint)
        encoders[i].eval()
        checkpoint = torch.load(os.path.join(pth, 'best_ensembles.pth'), map_location=torch.device('cpu'))
        mlps[i].load_state_dict(checkpoint)
        mlps[i].eval()

        # load scaler
        with open(os.path.join(pth, "scaler.pkl"), 'rb') as f:
            scaler = pickle.load(f)
            scalers.append(scaler)
    torch.set_grad_enabled(False)

    all_auroc = []
    all_auprc = []

    random_auroc = []
    random_auprc = []

    for patient in os.listdir(args.features_path):
        patient_dir = os.path.join(args.features_path, patient)

        user_relapse_labels = []
        user_anomaly_scores = []
        relapse_shapes = 0
        if patient == ".DS_Store":
            continue
        patient_id = int(patient[1:]) - 1
        if patient_id >= len(encoders):
            continue

        for subfolder in os.listdir(patient_dir):
            if (args.mode == 'val' and 'val' in subfolder) or (args.mode == 'test' and 'test' in subfolder):
                subfolder_dir = os.path.join(patient_dir, subfolder)
                file = 'features_stretched_w_steps.csv'
                file_path = os.path.join(subfolder_dir, file)
                df = pd.read_csv(file_path)  #, index_col=0)
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.dropna()

                try:
                    df[target_columns]
                except KeyError:
                    mins = df['mins']
                    sin_t, cos_t = calculate_sincos_from_minutes(mins)
                    df['sin_t'] = sin_t
                    df['cos_t'] = cos_t
            
                relapse_df = pd.read_csv(os.path.join(args.dataset_path, patient, subfolder, 'relapses.csv'))
                # IMPORTANT - drop last row as it was falsely added by the organizers
                relapse_df = relapse_df.iloc[:-1]

                # count 0 and 1 to calculate random chance
                if args.mode != 'test':
                    relapse_shapes += relapse_df['relapse'].sum()

                day_indices = relapse_df['day_index'].unique() # get all day indices for this patient

                for day_index in day_indices:
                    day_data = df[df['day_index'] == day_index].copy()

                    # relapse_label = relapse_df[relapse_df['day_index'] == day_index]['relapse'].to_numpy()[0]

                    if len(day_data) < args.window_size:
                        # predict zero anomaly score for days without enough data - right in the middle of the inlier/outlier of robust covariance
                        relapse_df.loc[relapse_df['day_index'] == day_index, 'anomaly_score'] = 0
                        user_anomaly_scores.append(0)
                        user_relapse_labels.append(0)
                        continue
                    
                    sequences = []
                    if len(day_data) == window_size:
                        sequence = day_data.iloc[0:window_size]
                        sequence = sequence[data_columns].copy().to_numpy()
                        sequence = scalers[patient_id].transform(sequence)
                        sequences.append(sequence)
                    else:
                        for start_idx in range(0, len(day_data) - window_size, window_size//3): # 1/3 overlap
                            sequence = day_data.iloc[start_idx:start_idx + window_size]
                            sequence = sequence[data_columns].copy().to_numpy()
                            sequence = scalers[patient_id].transform(sequence)
                            sequences.append(sequence)
                    sequence = np.stack(sequences)
                    sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
                    sequence_tensor = sequence_tensor.permute(0, 2, 1)


                    # Forward
                    features, _ = encoders[patient_id](sequence_tensor.to(device))
                    batched_features = features[None, :, :].repeat([args.ensembles, 1, 1])
                    # preds.shape = (ens_size, num_day_samples, output_dim)
                    preds = mlps[patient_id].forward(batched_features)
                    average_pred = torch.mean(preds, 0)

                    _mean = np.mean(train_dist_anomaly_scores[patient_id])
                    _max = np.max(train_dist_anomaly_scores[patient_id])
                    _min = np.min(train_dist_anomaly_scores[patient_id])

                    var_score = torch.sum((preds - average_pred) ** 2, dim=(2,))
                    mean_var = torch.mean(torch.mean(var_score, 0)).item()
                    anomaly_score = (mean_var - _mean) / (_max - _min)

                    anomaly_score = (np.array(anomaly_score.item()) > 0.0).astype(np.float64)

                    # add this to the relapse_df
                    relapse_df.loc[relapse_df['day_index'] == day_index, 'anomaly_score'] = anomaly_score
                    user_anomaly_scores.append(anomaly_score)
                    user_relapse_labels.append(0)

                # save subfolder in submission_path
                os.makedirs(os.path.join(args.submission_path, f'patient{patient[1]}', subfolder), exist_ok=True)
                csv_save_path = os.path.join(args.submission_path, f'patient{patient[1]}', subfolder, 'submission.csv')
                relapse_df.to_csv(csv_save_path, index = False)
                print(f"saved to: {csv_save_path}")

        user_anomaly_scores = np.array(user_anomaly_scores)
        user_relapse_labels = np.array(user_relapse_labels)

        # if mode is not test, calculate scores
        if args.mode != 'test':

            # create df per user if you want to see per user anomaly scores/relapses
            # user_df = pd.DataFrame()
            # user_df['anomaly_score'] = user_anomaly_scores
            # user_df['relapse'] = user_relapse_labels
            # # sort by anomaly score
            # user_df = user_df.sort_values(by=['anomaly_score'], ascending=False)
            # user_df.to_csv("{}.csv".format(patient), index=False)

            # Compute ROC Curve
            precision, recall, _ = sklearn.metrics.precision_recall_curve(user_relapse_labels, user_anomaly_scores)

            fpr, tpr, _ = sklearn.metrics.roc_curve(user_relapse_labels, user_anomaly_scores)

            # # Compute AUROC
            auroc = sklearn.metrics.auc(fpr, tpr)

            # # Compute AUPRC
            auprc = sklearn.metrics.auc(recall, precision)

            random_auroc.append(0.5)

            random_auprc.append(user_relapse_labels.mean())
            all_auroc.append(auroc)
            all_auprc.append(auprc)
            # auprc = pr_auc_score(user_relapse_labels, user_anomaly_scores)
            print(f'USER: {patient}, AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, Random AUPRC: {user_relapse_labels.mean():.4f}')

    if args.mode != 'test':

        total_auroc = sum(all_auroc)/len(all_auroc)
        total_auprc = sum(all_auprc)/len(all_auprc)
        random_auroc = sum(random_auroc)/len(random_auroc)
        random_auprc = sum(random_auprc)/len(random_auprc)
        total_avg = (total_auroc + total_auprc) / 2
        print(f'Total AUROC: {total_auroc:.4f}, Total AUPRC: {total_auprc:.4f}, Random AVG: {total_avg:.4f}, Random AUROC: {random_auroc:.4f}, Random AUPRC: {random_auprc:.4f}, Ideal AVG: {(random_auroc + random_auprc)/2:.4f}')


if __name__ == '__main__':
    main()
