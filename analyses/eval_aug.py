from ctgan import TVAE, CTGAN

# Load the best models
tvae_hard = TVAE.load('../outputs/synthesizers/tuning/0.pkl')
tvae_tot = TVAE.load('../outputs/synthesizers/tuning_tot/3.pkl')
ctgan_hard = TVAE.load('../outputs/synthesizers/tuning/3.pkl')
ctgan_tot = TVAE.load('../outputs/synthesizers/tuning_tot/1.pkl')

import sys, os
sys.path.append(os.path.join(os.getcwd(), '..'))
 
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm

# Import data and create DataFrame of data to augment
X_train = np.load('../data/processed/train.npz')['x']
y_train = np.load('../data/processed/train.npz')['y']
X_v = np.load('../data/processed/v.npz')['x']
y_v = np.load('../data/processed/v.npz')['y']

import json
import xgboost as xgb

from src.utils.amex_metric import *
from src.utils.train_xgb import *

tot_score, _ = train_xgb(X_train, y_train, X_v, y_v, n=10)

def get_xgboost_score(X_train, y_train):
    # Compute difference in average score with respect to all data
    new_score, _ = train_xgb(X_train, y_train, X_v, y_v, n=10)
    return new_score - tot_score

def get_score_augment(mod):
    synth = mod.sample(amount)
    X_synth = synth.drop('target', axis=1).values
    y_synth = np.array(synth['target'])
    X_train_aug = np.vstack((X_train, X_synth))
    y_train_aug = np.concatenate((y_train, y_synth))
    return get_xgboost_score(X_train_aug, y_train_aug)

agg_scores_tvae_hard = []
agg_scores_tvae_tot = []
agg_scores_ctgan_hard = []
agg_scores_ctgan_tot = []
five_perc = len(y_train) // 20
augment_amounts = [five_perc, five_perc * 2, five_perc * 3, five_perc * 4] # 5%, 10%, 15%, 20%
for amount in augment_amounts:
    scores_tvae_hard = []
    scores_tvae_tot = []
    scores_ctgan_hard = []
    scores_ctgan_tot = []
    for _ in tqdm(range(10)): # Repeat n times to mitigate randomness
        scores_tvae_hard.append(get_score_augment(tvae_hard))
        scores_tvae_tot.append(get_score_augment(tvae_tot))
        scores_ctgan_hard.append(get_score_augment(ctgan_hard))
        scores_ctgan_tot.append(get_score_augment(ctgan_tot))
    agg_scores_tvae_hard.append(scores_tvae_hard)
    agg_scores_tvae_tot.append(scores_tvae_tot)
    agg_scores_ctgan_hard.append(scores_ctgan_hard)
    agg_scores_ctgan_tot.append(scores_ctgan_tot)

with open('../outputs/results/aug_tvae_hard.pkl', 'wb') as f:
    pickle.dump(agg_scores_tvae_hard, f)
with open('../outputs/results/aug_tvae_tot.pkl', 'wb') as f:
    pickle.dump(agg_scores_tvae_tot, f)
with open('../outputs/results/aug_ctgan_hard.pkl', 'wb') as f:
    pickle.dump(agg_scores_ctgan_hard, f)
with open('../outputs/results/aug_ctgan_tot.pkl', 'wb') as f:
    pickle.dump(agg_scores_ctgan_tot, f)