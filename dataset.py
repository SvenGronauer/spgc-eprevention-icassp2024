import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'

class PatientDataset(Dataset):
    def __init__(self, features_path, dataset_path, patient, mode='train', scaler=None, window_size=48, stride=12):
        self.features_path = features_path
        self.dataset_path = dataset_path # to load relapses
        self.mode = mode
        self.window_size = window_size
        self.stride = stride

        self.columns_to_scale = ['acc_norm', 'gyr_norm', 'heartRate_mean', 'rRInterval_mean',
                                 'rRInterval_rmssd', 'rRInterval_sdnn', 'rRInterval_lombscargle_power_high', 'steps']
        self.data_columns = self.columns_to_scale + ['mins']

        self.data = []


        all_data = pd.DataFrame()
        patient_dir = os.path.join(features_path, patient)

        for subfolder in os.listdir(patient_dir):
            if ('train' in mode and 'train' in subfolder and subfolder.endswith('train')) or \
                    (mode == 'val' and 'val' in subfolder and subfolder.endswith('val')) or (mode == 'test' and 'test' in subfolder):
                subfolder_dir = os.path.join(patient_dir, subfolder)
                for file in os.listdir(subfolder_dir):
                    if file.endswith('features_stretched_w_steps.csv'):
                        file_path = os.path.join(subfolder_dir, file)
                        df = pd.read_csv(file_path)
                        df = df.replace([np.inf, -np.inf], np.nan)
                        for col in self.columns_to_scale:
                            for min in df['mins'].unique():
                                min_slice = df.loc[df['mins'] == min, col]
                                df.loc[(df[col].isna()) & (df['mins'] == min), col] = min_slice.median()
                        all_data = pd.concat([all_data, df])
                        day_indices = df['day'].unique()

                        if "train" not in mode:
                            relapse_df = pd.read_csv(os.path.join(self.features_path, patient, subfolder, 'relapse_stretched.csv'))

                        for day_index in day_indices:
                            day_data = df[df['day'] == day_index].copy()
                            if "train" not in mode:
                                try:
                                    relapse_label = relapse_df[relapse_df['day'] == day_index]['relapse'].values[0]
                                except:
                                    relapse_label = 0


                            if len(day_data) < self.window_size:
                                continue

                            if mode == "train":
                                # gather all data in this day with an overlap window of 12 (1H) and for duration of window_size
                                for start_idx in range(0, len(day_data) - self.window_size + 1, self.stride):
                                    sequence = day_data.iloc[start_idx:start_idx + self.window_size]
                                    sequence = sequence[self.data_columns].copy().to_numpy()
                                    self.data.append((sequence, 0, int(patient[-1])))
                            elif mode == "val":
                                # during validation we need all data to get all subsequences
                                self.data.append((day_data, relapse_label, int(patient[-1])))
                            elif mode == "train_dist":
                                # during validation we need all data to get all subsequences
                                self.data.append((day_data, 0, int(patient[-1])))
                             

        if scaler is None:
            print(mode, "fitting scaler")
            self.scaler = MinMaxScaler()
            self.scaler.fit(all_data[self.columns_to_scale].dropna().to_numpy())
        else:
            self.scaler = scaler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        day_data, relapse_label, patient_id = self.data[idx]
        if self.mode == 'train':
            sequence = day_data
            sequence[:, :-1] = self.scaler.transform(sequence[:, :-1]) # scale all columns except min
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
            sequence_tensor = sequence_tensor.permute(1, 0)

        else: 
            sequences = []
            if len(day_data) < self.window_size:
                print("Day data is less than window size")
                # Handle accordingly
                return None 
            
            if len(day_data) == self.window_size:
                start_idx = 0
                sequence = day_data.iloc[start_idx:start_idx + self.window_size]
                sequence = sequence[self.data_columns].copy().to_numpy()
                sequence[:, :-1] = self.scaler.transform(sequence[:, :-1])
                sequences.append(sequence)
            else:
                for start_idx in range(0, len(day_data) - self.window_size + 1, self.stride):
                    sequence = day_data.iloc[start_idx:start_idx + self.window_size]
                    sequence = sequence[self.data_columns].copy().to_numpy()
                    sequence[:, :-1] = self.scaler.transform(sequence[:, :-1])
                    sequences.append(sequence)
            sequence = np.stack(sequences)
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
            sequence_tensor = sequence_tensor.permute(0, 2, 1)
        
        return {
            'data': sequence_tensor,
            'user_id': torch.tensor(patient_id, dtype=torch.long)-1,
            'relapse_label': torch.tensor(relapse_label, dtype=torch.long),
        }


