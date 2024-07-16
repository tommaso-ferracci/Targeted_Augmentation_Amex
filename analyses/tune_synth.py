import sys, os
sys.path.append(os.path.join(os.getcwd(), '..'))

import copy
import pickle
import numpy as np
import pandas as pd

from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer, CTGANSynthesizer
from baytune import BTBSession
from baytune.tuning import Tunable
from baytune.tuning import hyperparams as hp

from src.utils.train_xgb import train_xgb

def append_to_pickle_file(pickle_file, new_data):
    try:
        with open(pickle_file, 'rb') as f:
            existing_data = pickle.load(f)
        if not isinstance(existing_data, list):
            existing_data = [existing_data]
        existing_data.append(new_data)
        with open(pickle_file, 'wb') as f:
            pickle.dump(existing_data, f)
    except FileNotFoundError:
        with open(pickle_file, 'wb') as f:
            pickle.dump([new_data], f)

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

# Compute average score on all data
tot_score, _ = train_xgb(X_train, y_train, X_v, y_v, n=10)
print(tot_score)

def get_xgboost_score(X_train, y_train):
    # Compute difference in average score with respect to all data
    new_score, _ = train_xgb(X_train, y_train, X_v, y_v, n=10)
    return new_score - tot_score

mods = {
    'TVAE': TVAESynthesizer,
    'CTGAN': CTGANSynthesizer,
}

def scoring_function(mod_name, hyperparams):
    '''
    Scorer for a synthesizer: quantifies validation performance variation 
    when synthetic data is added to training dataset.
    '''
    mod_class = mods[mod_name]
    mod_instance = mod_class(metadata, **hyperparams)
    mod_instance.fit(X_worst_df)
    scores = []
    for _ in range(10): # Repeat 10 times to mitigate randomness
        synthetic_data = mod_instance.sample(num_rows=len(y_worst))
        X_synth = synthetic_data.drop('target', axis=1).values
        y_synth = np.array(synthetic_data['target'])
        X_train_aug = np.vstack((X_train, X_synth))
        y_train_aug = np.concatenate((y_train, y_synth))
        scores.append(get_xgboost_score(X_train_aug, y_train_aug))
    hyperparams_log = copy.deepcopy(hyperparams)
    hyperparams_log['mod_name'] = mod_name
    append_to_pickle_file('../outputs/synthesizers/hyperparams_hard_10.pkl', hyperparams_log)
    append_to_pickle_file('../outputs/synthesizers/scores_hard_10.pkl', scores)
    return np.mean(scores)

# Candidate models and their hyperparameter sets
tunables = {
    'TVAE': Tunable({
    'batch_size': hp.IntHyperParam(min=100, max=1000, default=500, step=1),
    'epochs': hp.IntHyperParam(min=50, max=300, default=100, step=1),
    'embedding_dim': hp.IntHyperParam(min=64, max=512,default=128, step=1),
}),
    'CTGAN': Tunable({
    'batch_size': hp.IntHyperParam(min=100, max=1000,default=500, step=1),
    'epochs': hp.IntHyperParam(min=20, max=200, default=50, step=1),
    'embedding_dim': hp.IntHyperParam(min=64, max=512,default=128, step=1),
})
}

session = BTBSession(
    tunables=tunables,
    scorer=scoring_function,
    verbose=True
)

best_prop = session.run(50)
print(best_prop)

# Dump session results
with open('../outputs/synthesizers/session_hard_10.pkl', "wb") as f:
    pickle.dump(session, f)
    