import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os


def calculate_time_from_sincos(sin_t, cos_t):
    angle = np.arctan2(sin_t, cos_t)
    time_value = (angle + 2. * np.pi) % (2. * np.pi)
    minutes = (time_value / (2. * np.pi / (60 * 24)))
    return minutes


def stretch_features(patient, phase, processed_feats):
    df = pd.read_csv(f'{processed_feats}/{patient}/{phase}/features.csv')
    df = df.replace([np.inf, -np.inf], np.nan)
    df.fillna(0, inplace=True)
    df['time_value'] = calculate_time_from_sincos(df['sin_t'], df['cos_t'])
    new_df = {}
    new_df['mins'] = []
    new_df['day'] = []
    for col_i, col in enumerate(['acc_norm', 'heartRate_mean', 'rRInterval_mean', 'rRInterval_rmssd',
                                 'rRInterval_sdnn', 'rRInterval_lombscargle_power_high', 'gyr_norm']):
        new_df[col] = []
        to_plot = pd.pivot_table(df.reset_index(),
                                 index='time_value', columns='day_index', values=col)
        axes = to_plot.plot(subplots=True)
        for i_ax, ax in enumerate(axes):
            plt.setp(ax, xlim=(0, 60 * 24))
            y_vals = ax.lines[0].get_ydata().tolist()
            new_df[col].extend(y_vals)
            if col_i == 0:
                x_vals = ax.lines[0].get_xdata().tolist()
                new_df['mins'].extend(x_vals)
                day_label = ax.lines[0].get_label()
                new_df['day'].extend([day_label for _ in range(len(y_vals))])
        plt.clf()
        plt.close()
    new_df = pd.DataFrame(new_df)
    new_df = new_df.rename(columns={'day': 'day_index'})
    new_df.to_csv(f'{processed_feats}/{patient}/{phase}/features_stretched.csv', index=False)


def reformat_time_cols(df):
    df['start_time'] = pd.to_datetime(df['start_time'], format="%H:%M:%S")
    df['end_time'] = pd.to_datetime(df['end_time'], format="%H:%M:%S")
    df['start_time'] = (
            df['start_time'].dt.hour * 60 + df['start_time'].dt.minute + df['start_time'].dt.second / 60).round()
    df['end_time'] = (df['end_time'].dt.hour * 60 + df['end_time'].dt.minute + df['end_time'].dt.second / 60).round()
    df['diff'] = df['end_time'] - df['start_time']
    return df


def add_steps(patient, phase, raw_data, processed_feats):
    df = pd.read_parquet(f'{raw_data}/{patient}/{phase}/step.parquet')
    df = reformat_time_cols(df)
    df = df.replace([np.inf, -np.inf], 0)
    features_stretched = pd.read_csv(f'{processed_feats}/{patient}/{phase}/features_stretched.csv')
    day_idxs = sorted(features_stretched['day_index'].unique().tolist())
    day_steps = np.zeros((len(day_idxs), 60 * 24))
    accum = np.zeros((len(day_idxs), 60 // 5 * 24))
    for day_idx, day in enumerate(day_idxs):
        df_day = df[(df['start_date_index'] == day) & (df['end_date_index'] == day)]
        for row_i, row in df_day.iterrows():
            if (row['totalSteps'] >= 0):
                if row['start_date_index'] == row['end_date_index']:
                    if row['diff'] == 0:
                        day_steps[day_idx, min(60 * 24 - 1, row['start_time'].astype(int))] = row['totalSteps']
                    else:
                        partial_steps = row['totalSteps'] / (row['diff'])
                        for i in range(row['start_time'].astype(int), row['end_time'].astype(int)):
                            day_steps[day_idx, i] = partial_steps
        accum[day_idx] = np.array([day_steps[day_idx, x:x + 5].sum() for x in range(0, 60 * 24, 5)])
    features_stretched['steps'] = accum.reshape(-1)
    features_stretched.to_csv(f'{processed_feats}/{patient}/{phase}/features_stretched_w_steps.csv', index=False)


def combine_files(dataset_path, processed_feats, phase, patient, file_name):
    all_df = pd.DataFrame()
    max_day_so_far = 0
    for i in range(3):
        if os.path.isdir(f'{dataset_path}/{patient}/{phase}_{str(i)}'):
            df = pd.read_csv(f'{dataset_path}/{patient}/{phase}_{str(i)}/{file_name}.csv')
            df['day'] = df['day_index'] + max_day_so_far
            all_df = pd.concat([all_df, df])
            max_day_so_far += df['day_index'].max() + 1
    all_df = all_df.replace([np.inf, -np.inf], np.nan)
    all_df.fillna(0)
    if not os.path.exists(f'{processed_feats}/{patient}/{phase}'):
        os.mkdir(f'{processed_feats}/{patient}/{phase}')
    all_df.to_csv(f'{processed_feats}/{patient}/{phase}/{file_name}.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dataset_path', type=str,
                        default='/media/storage/ge24bud/spgc-eprevention-icassp2024/track_1',
                        help='path to raw downloaded data')
    parser.add_argument('--preprocessed_data_path', type=str,
                        default='/media/storage/ge24bud/spgc-eprevention-icassp2024/track_1_features',
                        help='path to preprocessed data')

    args = parser.parse_args()

    patients = os.listdir(args.raw_dataset_path)
    combs = []
    for patient in patients:
        if os.path.isdir(os.path.join(args.raw_dataset_path, patient)):
            combine_files(args.raw_dataset_path, args.preprocessed_data_path, 'val', patient, 'relapses')
            for phase in os.listdir(os.path.join(args.raw_dataset_path, patient)):
                if 'DS_Store' not in phase:
                    combs.append([patient, phase])

    for patient, phase in combs:
        print(patient, phase)
        stretch_features(patient, phase, args.preprocessed_data_path)
        add_steps(patient, phase, args.raw_dataset_path, args.preprocessed_data_path)

    for patient in ['P4']:
        combine_files(args.preprocessed_data_path, args.preprocessed_data_path, 'train', patient,
                      'features_stretched_w_steps')
        combine_files(args.preprocessed_data_path, args.preprocessed_data_path, 'val', patient,
                      'features_stretched_w_steps')
