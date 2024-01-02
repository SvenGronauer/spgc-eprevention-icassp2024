from copy import copy
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
import re
from pathlib import Path

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='directory with files to merge')  # e.g. "/var/tmp/spgc/track1
    args = parser.parse_args()
    return args

def replace_string_at_index(s: str, position: int, replacement: str, chars_to_replace: int = 1):
    s = s[:position] + replacement + s[position + chars_to_replace:]
    return s

def open_csv_files(file_path: str, num_runs: int):
    data_frames = []
    idx = re.search("run", file_path).start()
    asdf = file_path[idx:idx+3]
    for i in range(num_runs):
        fnp = replace_string_at_index(file_path, idx+3, str(i+1))
        df = pd.read_csv(fnp)
        data_frames.append(df)
    return data_frames

def main():
    args = parse()
    dirs = os.listdir(args.path)
    dirs = [d for d in dirs if (d != ".DS_Store" and d != "merged")]
    print(dirs)
    merge_path = os.path.join(args.path, "merged")
    # for directory in dirs:
    #
    #     print(directory)
    for path, directories, files in os.walk(os.path.join(args.path, dirs[0])):
        for file in files:
            if file == ".DS_Store":
                continue
            print(f' {path} file: {file}')
            # fnp == '/var/tmp/spgc/track1/run2/patient6/test_0/submission.csv'
            fnp = os.path.join(path, file)
            data_frames = open_csv_files(fnp, num_runs=len(dirs))
            averages = pd.concat([each.stack() for each in data_frames], axis=1) \
                .apply(lambda x: x.mean(), axis=1) \
                .unstack()
            a = 5

            idx = re.search("run", fnp).start()
            merged_csv_path = replace_string_at_index(fnp, idx, 'merged', chars_to_replace=4)  # replace runX with merged
            output_file_path = Path(merged_csv_path)
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            averages.to_csv(merged_csv_path, index = False)
            print(f"-- saved to: {merged_csv_path}")

if __name__ == '__main__':
    main()
