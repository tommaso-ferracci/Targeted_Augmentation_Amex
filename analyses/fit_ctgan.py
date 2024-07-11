import sys, os
sys.path.append(os.path.join(os.getcwd(), '..'))

import numpy as np
import pandas as pd

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# Import data and create DataFrame of data to augment
X_train = np.load('../data/processed/train.npz')['x']
y_train = np.load('../data/processed/train.npz')['y']
X_v = np.load('../data/processed/v.npz')['x']
y_v = np.load('../data/processed/v.npz')['y']

knn_5 = pd.read_csv('../outputs/results/knn_5.csv')['0']
ind_5 = knn_5.sort_values(ascending=True).index
X_worst = X_train[ind_5[:(len(X_train) // 10)]]
y_worst = y_train[ind_5[:(len(X_train) // 10)]]

X_worst_df = pd.DataFrame(X_worst)
X_worst_df['target'] = y_worst

# Infer metadata: categorical and numeric features
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(X_worst_df)

synthesizer = CTGANSynthesizer(
    metadata, # required
    batch_size=100,
    epochs=21,
    embedding_dim=64,
    cuda='cuda:5',
    verbose=True
)

synthesizer.fit(X_worst_df)

synthesizer.save(
    filepath='../outputs/synthesizers/best_ctgan.pkl'
)
