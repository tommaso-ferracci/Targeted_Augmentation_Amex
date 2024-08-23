import sys, os
sys.path.append(os.path.join(os.getcwd(), '..'))

import copy
import pickle
import numpy as np
import pandas as pd

from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer, CTGANSynthesizer
from ctgan import TVAE, CTGAN
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

knn_100 = pd.read_csv('../outputs/results/knn_100.csv')['0']
ind_100 = knn_100.sort_values(ascending=True).index
X_hard = X_train[ind_100[:(len(X_train) // 10)]]
y_hard = y_train[ind_100[:(len(X_train) // 10)]]

X_hard_df = pd.DataFrame(X_hard)
X_hard_df['target'] = y_hard

# Infer metadata: categorical and numeric features
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(X_hard_df)

# Compute average score on all data
tot_score, _ = train_xgb(X_train, y_train, X_v, y_v, n=100)
print(tot_score)

def get_xgboost_score(X_train, y_train):
    # Compute difference in average score with respect to all data
    new_score, _ = train_xgb(X_train, y_train, X_v, y_v, n=100)
    return new_score - tot_score

mods = {
    'TVAE': TVAESynthesizer,
    'CTGAN': CTGANSynthesizer,
}

synths = {
    'TVAE': TVAE,
    'CTGAN': CTGAN
}

def transform_dict(mod_name, hyperparams):
    params = {}
    if mod_name == 'TVAE':
        params['embedding_dim'] = hyperparams['embedding_dim']
        params['compress_dims'] = (hyperparams['compress_dims_0'], hyperparams['compress_dims_1'])
        params['decompress_dims'] = (hyperparams['decompress_dims_0'], hyperparams['decompress_dims_1'])
    else:
        params['embedding_dim'] = hyperparams['embedding_dim']
        params['discriminator_dim'] = (hyperparams['discriminator_dim_0'], hyperparams['discriminator_dim_1'])
        params['generator_dim'] = (hyperparams['generator_dim_0'], hyperparams['generator_dim_1'])
    return params

def scoring_function(mod_name, hyperparams):
    '''
    Scorer for a synthesizer: quantifies validation performance variation 
    when synthetic data is added to training dataset.
    '''
    global ind
    mod_class = mods[mod_name]
    params = transform_dict(mod_name, hyperparams)
    mod_instance = mod_class(metadata, 
                             save_path=f'../outputs/synthesizers/tuning_2/{ind}.pkl',
                             epochs=500,
                             batch_size=10000,
                             cuda='cuda:1',
                             patience=50,
                             weights=pd.read_csv('../outputs/results/feature_importances.csv')['gain'].values,
                             **params)
    mod_instance.fit(X_hard_df)
    # restore best model
    mod_instance = synths[mod_name].load(f'../outputs/synthesizers/tuning_2/{ind}.pkl')
    ind += 1
    scores = []
    for _ in range(10): # Repeat 10 times to mitigate randomness
        synthetic_data = mod_instance.sample(len(y_hard))
        X_synth = synthetic_data.drop('target', axis=1).values
        y_synth = np.array(synthetic_data['target'])
        X_train_aug = np.vstack((X_train, X_synth))
        y_train_aug = np.concatenate((y_train, y_synth))
        scores.append(get_xgboost_score(X_train_aug, y_train_aug))
    hyperparams_log = copy.deepcopy(hyperparams)
    hyperparams_log['mod_name'] = mod_name
    append_to_pickle_file('../outputs/synthesizers/hyperparams_hard_2.pkl', hyperparams_log)
    append_to_pickle_file('../outputs/synthesizers/scores_hard_2.pkl', scores)
    return np.mean(scores)

# Candidate models and their hyperparameter sets
tunables = {
    'TVAE': Tunable({
    'embedding_dim': hp.IntHyperParam(min=32, max=512, default=64, step=1),
    'compress_dims_0': hp.IntHyperParam(min=32, max=512, default=128, step=1), 
    'compress_dims_1': hp.IntHyperParam(min=32, max=512, default=128, step=1),
    'decompress_dims_0': hp.IntHyperParam(min=32, max=512, default=128, step=1), 
    'decompress_dims_1': hp.IntHyperParam(min=32, max=512, default=128, step=1),
}),
    'CTGAN': Tunable({
    'embedding_dim': hp.IntHyperParam(min=32, max=512, default=64, step=1),
    'discriminator_dim_0': hp.IntHyperParam(min=32, max=512, default=64, step=1), 
    'discriminator_dim_1': hp.IntHyperParam(min=32, max=512, default=64, step=1),
    'generator_dim_0': hp.IntHyperParam(min=32, max=512, default=64, step=1), 
    'generator_dim_1': hp.IntHyperParam(min=32, max=512, default=64, step=1),
})
}

session = BTBSession(
    tunables=tunables,
    scorer=scoring_function,
    verbose=True
)

ind = 0
best_prop = session.run(20)
print(best_prop)

# Dump session results
with open('../outputs/synthesizers/session_hard_2.pkl', "wb") as f:
    pickle.dump(session, f)
    