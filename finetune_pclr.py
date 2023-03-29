import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.utils import resample

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import Adam

from ecg import ECGDataset
from config import args


def get_model(embedding_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> Model:
    """Get PCLR embedding model and finetune it"""
    # Load PCLR model
    base_model = load_model("./PCLR.h5")
    base_model.trainable = False

    # Create finetune classifier model
    inputs = Input(shape=(4096, 12))
    x = base_model(inputs, training=False)
    x = Dense(hidden_dim, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    # Combine base model and classifier model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def get_data(args, batch_size):
    df_tab = pd.read_csv(os.path.join(args.dir_csv, "tabular_data.csv"))
    train_ids = np.load("./stores/train_ids.npy")
    val_ids = np.load("./stores/val_ids.npy")
    test_ids = np.load("./stores/test_ids.npy")

    train_ids = train_ids[len(train_ids) // 2 :]
    val_ids = val_ids[len(val_ids) // 2 :]
    test_ids = test_ids[len(test_ids) // 2 :]

    train_df = df_tab[df_tab["QuantaID"].isin(train_ids)]
    val_df = df_tab[df_tab["QuantaID"].isin(val_ids)]
    test_df = df_tab[df_tab["QuantaID"].isin(test_ids)]
    print(len(train_df), len(val_df), len(test_df))

    train_dataset = ECGDataset(args, train_df)
    val_dataset = ECGDataset(args, val_df)
    test_dataset = ECGDataset(args, test_df)

    train_generator = train_dataset.load_ecg_dataset(args.batch_size)
    val_generator = val_dataset.load_ecg_dataset(batch_size)

    return train_generator, val_generator, train_dataset, val_dataset, test_dataset

def bootstrap_evaluation(test_dataset, model, batch_size, num_bootstrap_samples):
    test_accuracies = []

    for i in range(num_bootstrap_samples):
        # Resample test dataset with replacement
        resampled_test_dataset = resample(test_dataset, replace=True)

        # Create a generator for the resampled test dataset
        resampled_test_generator = resampled_test_dataset.load_ecg_dataset(batch_size)

        # Evaluate the model on the resampled test dataset
        _, test_accuracy = model.evaluate(
            resampled_test_generator, steps=len(resampled_test_dataset) // batch_size, verbose=0
        )
        
        test_accuracies.append(test_accuracy)

    return test_accuracies

# Load ECG dataset
train_generator, val_generator, train_dataset, val_dataset, test_dataset = get_data(args, args.batch_size)


# Manually iterate through the generator
try:
    for i, (inputs, labels) in enumerate(train_generator):
        print(f"Iteration {i}: inputs shape = {inputs.shape}, labels shape = {labels.shape}")
        if i > 10:  # Limit the number of iterations to avoid an infinite loop
            break
except Exception as e:
    print(f"Error: {e}")


model = get_model(
    embedding_dim=args.embedding_dim,
    hidden_dim=args.hidden_dim,
    num_classes=args.num_classes,
    dropout=args.dropout,
)

# Finetune the model on the ECG dataset
model.fit(
    train_generator,
    epochs=args.epochs,
    steps_per_epoch=len(train_dataset) // args.batch_size,
    validation_data=val_generator,
    validation_steps=len(val_dataset) // args.batch_size,
)

# Perform bootstrapping for testing
num_bootstrap_samples = 1000
test_accuracies = bootstrap_evaluation(test_dataset, model, args.batch_size, num_bootstrap_samples)

# Calculate the mean and standard deviation of the test accuracies
mean_test_accuracy = np.mean(test_accuracies)
std_test_accuracy = np.std(test_accuracies)

print(f"Mean test accuracy: {mean_test_accuracy:.4f}")
print(f"Standard deviation of test accuracy: {std_test_accuracy:.4f}")

# Calculate the 95% confidence interval for the test accuracies
lower_bound = np.percentile(test_accuracies, 2.5)
upper_bound = np.percentile(test_accuracies, 97.5)

print(f"95% confidence interval for test accuracy: ({lower_bound:.4f}, {upper_bound:.4f})")