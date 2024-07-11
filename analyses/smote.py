import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from imblearn.over_sampling import SMOTE

import sys, os
sys.path.append(os.path.join(os.getcwd(), '..'))

from src.utils.train_xgb import train_xgb

X_train = np.load('../data/processed/train.npz', allow_pickle=True)['x']
y_train = np.load('../data/processed/train.npz', allow_pickle=True)['y']
X_v = np.load('../data/processed/v.npz', allow_pickle=True)['x']
y_v = np.load('../data/processed/v.npz', allow_pickle=True)['y']

current_class_distribution = pd.Series(y_train).value_counts()
majority_class = current_class_distribution.idxmax()
minority_class = current_class_distribution.idxmin()

sampling_strategy = {
    majority_class: int(current_class_distribution[majority_class] * 1.1),
    minority_class: int(current_class_distribution[minority_class] * 1.1)
}

means_base = []
for _ in tqdm(range(10)):
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=_) 
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    mean, _ = train_xgb(X_smote, y_smote, X_v, y_v, n=10)
    means_base.append(mean)

with open('../outputs/synthesizers/smote_base.pkl', 'wb') as f:
    pickle.dump(means_base, f)

knn_5 = pd.read_csv('../outputs/results/knn_5.csv')['0']
ind_knn_5 = list(knn_5.sort_values(ascending=True).index)
X_hard = X_train[ind_knn_5[:(len(y_train) // 10)]]
y_hard = y_train[ind_knn_5[:(len(y_train) // 10)]]
X_rest = X_train[ind_knn_5[(len(y_train) // 10):]]
y_rest = y_train[ind_knn_5[(len(y_train) // 10):]]

current_class_distribution = pd.Series(y_hard).value_counts()
majority_class = current_class_distribution.idxmax()
minority_class = current_class_distribution.idxmin()

sampling_strategy = {
    majority_class: int(current_class_distribution[majority_class] * 2),
    minority_class: int(current_class_distribution[minority_class] * 2)
}

means_hard = []
for _ in tqdm(range(10)):
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=_) 
    X_smote_hard, y_smote_hard = smote.fit_resample(X_hard, y_hard)
    X_smote_hard = np.vstack((X_rest, X_smote_hard))
    y_smote_hard = np.concatenate((y_rest, y_smote_hard))
    mean, _ = train_xgb(X_smote_hard, y_smote_hard, X_v, y_v, n=10)
    means_hard.append(mean)

with open('../outputs/synthesizers/smote_hard.pkl', 'wb') as f:
    pickle.dump(means_hard, f)
