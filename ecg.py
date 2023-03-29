import os
import numpy as np
import pandas as pd

from tensorflow.keras.utils import to_categorical

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


class ECGDataset:
    def __init__(self, args, df, df_demo=None):  # , augment=False):
        # self.augment = augment
        self.args = args
        self.df = df
        self.df_demo = df_demo
        self.pcwp_train = np.load("./stores/train_info.npy")
        self.pcwp_mean = self.pcwp_train[0]
        self.pcwp_std = self.pcwp_train[1]

    def __len__(self):
        return len(self.df)

    def load_ecg_dataset(self, batch_size):
        num_batches = len(self.df) // batch_size

        while True:
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = start + batch_size
                batch_data = []

                for idx in range(start, end):
                    row = self.df.iloc[idx]
                    qid = row['QuantaID']
                    doc = row['Date_of_Cath']
                    fname = os.path.join(self.args.dir_csv, f'{qid}_{doc}.csv')

                    x = pd.read_csv(fname).values[...,1:].astype(np.float32)

                    if self.args.label == 'pcwp':
                        if self.args.train_mode == 'regression':
                            y = row['PCWP_mean']
                            if self.args.normalize_label:
                                y = (y - self.pcwp_mean) / (self.pcwp_std)
                        else:
                            y = row['PCWP_mean'] > self.args.pcwp_th
                    elif self.args.label == 'age':
                        if self.args.train_mode == 'regression':
                            y = row['Age_at_Cath']  # regression
                        else:
                            y = row['PCWP_mean'] > self.args.pcwp_th

                    elif self.args.label == 'gender':
                        y = row['Sex']

                    x = x / 1000
                    sample = (x[:2496, :].T, y)
                    batch_data.append(sample)

                inputs, labels = zip(*batch_data)
                inputs = np.stack(inputs)
                labels = to_categorical(labels)
                yield inputs, labels