"""
Author: Hyewon Jeong
Last Modified: January 20, 2∂22
"""

import numpy as np
import os
import sys
import pickle
import pandas as pd
import argparse
import tensorflow as tf

sys.path.append("../")

from src.preprocess_ecg import process_ecg, LEADS
import src.get_representations import get_representations

parser = argparse.ArgumentParser(description='ECG Aug Baseline')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()

tf.random.set_seed(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
    
tab = '/storage/shared/apollo/same-day/tabular_data.csv'
df_tab = pd.read_csv(tab)
df_tab = df_tab.dropna(subset=['CO'])
frac_train = 0.8
frac_val = 0.2

uids = df_tab['QuantaID'].unique()

def get_data(df):
    ecgs = []
    for idx in df.index:
        row = df.loc[idx]
        qid = row['QuantaID']
        doc = row['Date_of_Cath']
        fname = f'/storage/shared/apollo/same-day/{qid}_{doc}.csv'
        x = pd.read_csv(fname).values[...,1:].astype(np.float32)
        ecgs.append(x)
    ecgs = np.array(ecgs)
    ecgs /= 1000
    return np.transpose(ecgs, (0,2,1))

ecgs = get_data(df_tab)

test = get_representations(ecgs)
print(test.shape)

# Save this info
with open('pclr.npy', 'wb') as f:
    np.save(f, test)


