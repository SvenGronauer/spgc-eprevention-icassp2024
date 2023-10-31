import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np



class PatientDataset(Dataset):
    def __init__(self, features_path, dataset_path, mode='train', scaler=None, window_size=48):
        self.features_path = features_path
        self.dataset_path = dataset_path # to load relapses
        self.mode = mode
        self.window_size = window_size
        
        self.columns_to_scale = ['acc_norm', 'heartRate_mean', 'rRInterval_mean', 'rRInterval_rmssd', 'rRInterval_sdnn', 'rRInterval_lombscargle_power_high']
        self.data_columns = self.columns_to_scale + ['sin_t', 'cos_t']


        self.data = []


        for patient in sorted(os.listdir(features_path)):
            all_data = pd.DataFrame()
            patient_dir = os.path.join(features_path, patient)

            for subfolder in os.listdir(patient_dir):
                if (mode == 'train' and 'train' in subfolder) or (mode == 'val' and 'val' in subfolder) or (mode == 'test' and 'test' in subfolder):
                    subfolder_dir = os.path.join(patient_dir, subfolder)
                    for file in os.listdir(subfolder_dir):
                        if file.endswith('.csv'):
                            file_path = os.path.join(subfolder_dir, file)
                            df = pd.read_csv(file_path, index_col=0)
                            df = df.replace([np.inf, -np.inf], np.nan)
                            df = df.dropna() # something better could be used here - e.g. imputing
                       
                            all_data = pd.concat([all_data, df]) 

                            day_indices = df['day_index'].unique()

                            relapse_df = pd.read_csv(os.path.join(self.dataset_path, patient, subfolder, 'relapses.csv'))

                            for day_index in day_indices:
                                day_data = df[df['day_index'] == day_index].copy()

                                relapse_label = relapse_df[relapse_df['day_index'] == day_index]['relapse'].values[0]


                                if len(day_data) < self.window_size:
                                    continue
                                # columns in sequence:
                                # day_index  acc_norm  heartRate_mean  rRInterval_mean, rRInterval_sdnn, rRInterval_lombscargle,  gyr_norm     sin_t     cos_t
                                if mode == "train":
                                    # gather all data in this day with an overlap window of 12 (1H) and for duration of window_size  
                                    for start_idx in range(0, len(day_data) - self.window_size, 12): 
                                        # sequence is of size: (window_size-1, input_features)
                                        sequence = day_data.iloc[start_idx:start_idx+self.window_size-1]
                                        sequence = sequence[self.data_columns].copy().to_numpy()
                                        # target is of size: (1, input_features)
                                        last_entry = slice(start_idx+self.window_size-1, start_idx+self.window_size)
                                        target = day_data.iloc[last_entry]
                                        target = target[self.data_columns].copy().to_numpy()
                                        self.data.append((sequence, target, int(patient[1:]), relapse_label))
                                else:
                                    # during validation we need all data to get all subsequences
                                    self.data.append((day_data, None, int(patient[1:]), relapse_label))
                             


        if scaler is None:
            print(mode, "fitting scaler")
            self.scaler = MinMaxScaler()
            self.scaler.fit(all_data[self.columns_to_scale].dropna().to_numpy())
        else:
            self.scaler = scaler        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        day_data, target, patient_id, relapse_label = self.data[idx]
        if self.mode == 'train':
            sequence = day_data

            sequence[:, :-2] = self.scaler.transform(sequence[:, :-2]) # scale all columns except sin_t and cos_t
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
            sequence_tensor = sequence_tensor.permute(1, 0)

            target[:, :-2] = self.scaler.transform(target[:, :-2])
            target_tensor = torch.tensor(target, dtype=torch.float32)
            target_tensor = target_tensor.permute(1, 0)
        else:  # validation
            sequences = []
            targets = []
            if len(day_data) < self.window_size:
                print("Day data is less than window size")
                # Handle accordingly
                return None 
            
            if len(day_data) == self.window_size:
                sequence = day_data.iloc[0:self.window_size-1]
                sequence = sequence[self.data_columns].copy().to_numpy()
                sequence[:, :-2] = self.scaler.transform(sequence[:, :-2])
                sequences.append(sequence)

                last_entry = slice(self.window_size - 1, self.window_size)
                target = day_data.iloc[last_entry]
                target = target[self.data_columns].copy().to_numpy()
                target[:, :-2] = self.scaler.transform(target[:, :-2])
                targets.append(target)
            else:
                for start_idx in range(0, len(day_data) - self.window_size, self.window_size//3): # 1/3 overlap
                    sequence = day_data.iloc[start_idx:start_idx + self.window_size - 1]
                    sequence = sequence[self.data_columns].copy().to_numpy()
                    sequence[:, :-2] = self.scaler.transform(sequence[:, :-2])
                    sequences.append(sequence)

                    last_entry = slice(start_idx + self.window_size - 1,  start_idx + self.window_size)
                    target = day_data.iloc[last_entry]
                    target = target[self.data_columns].copy().to_numpy()
                    target[:, :-2] = self.scaler.transform(target[:, :-2])
                    targets.append(target)

            sequence = np.stack(sequences)
            target = np.stack(targets)
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
            sequence_tensor = sequence_tensor.permute(0, 2, 1)

            target_tensor = torch.tensor(target, dtype=torch.float32)
            target_tensor = target_tensor.permute(0, 2, 1)
        
        return {
            'data': sequence_tensor,
            'target': target_tensor,
            'user_id': torch.tensor(patient_id, dtype=torch.long)-1,
            'relapse_label': torch.tensor(relapse_label, dtype=torch.long),
        }


