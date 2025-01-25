import sys, os, json, time, copy, pickle, warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from ucimlrepo import fetch_ucirepo 
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer, CTGANSynthesizer
from ctgan import TVAE, CTGAN
from baytune import BTBSession
from baytune.tuning import Tunable
from baytune.tuning import hyperparams as hp

sys.path.append(os.path.join(os.getcwd(), '..'))
warnings.filterwarnings("ignore")
from src.utils.data_valuation import *
from src.utils.amex_metric import *
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

def run_targeted_aug(X, y, dataset_name, needs_encoding=False, k=5, n_retrain=3, n_retry=10, n_tune=20):
    '''
    Function to run the entire pipeline: 
    - n_retrain refers to how many times xgboost is retrained to mitigate randomness
    - n_retry refers to how many times we sample from the synthesisers to mitigate randomness
    - n_tune refers to the number of iterations of the hyperparameter tuning pipeline
    '''
    print("Status: preprocessing data...")
    X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test, X_v, y_test, y_v = train_test_split(X_, y_, test_size=0.5, random_state=42, stratify=y_)
    if needs_encoding:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        oe.fit(X_train)
        X_train = oe.transform(X_train)
        X_test = oe.transform(X_test)
        X_v = oe.transform(X_v)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    X_v = sc.transform(X_v)
    print("Status: data ready.")

    print("Status: starting KNN Shapleys calculation...")
    start_knn = time.time()
    knn = compute_knn_shapley(X_train, y_train, X_test, y_test, k=5)
    ind_knn = pd.Series(knn).sort_values(ascending=True).index
    end_knn = time.time()
    os.mkdir(f"../outputs/results/{dataset_name}")
    with open(f"../outputs/results/{dataset_name}/notes", "w") as f:
        f.write(f"Seconds to calculate KNN Shapleys: {end_knn - start_knn}.\n")
    print("Status: done with KNN Shapleys calculation.")

    print("Status: train baseline xgboost and calculate feature importances...")
    with open('../config/xgboost.json', 'r') as f:
        params = json.load(f)
    dtrain = xgb.DMatrix(X_train, y_train)
    dv = xgb.DMatrix(X_v, y_v)
    bst = xgb.train(params, dtrain, num_boost_round=9999, verbose_eval=0,
                    evals=[(dtrain, 'train'), (dv, 'v')], custom_metric=amex_scorer, 
                    early_stopping_rounds=100, maximize=True)
    baseline_score = amex_metric(y_v, bst.predict(dv, iteration_range=(0, bst.best_iteration + 1)))
    dict_gain = bst.get_score(importance_type='gain')
    dict_weight = bst.get_score(importance_type='weight')
    for ix in range(X_train.shape[1]):
        key = f'f{ix}'
        if key not in dict_gain:
            dict_gain[key] = 0
            dict_weight[key] = 0
    df_importance = pd.DataFrame({'feature': dict_gain.keys(),
                                  'gain': dict_gain.values(),
                                  'weight': dict_weight.values()})
    print(f"Status: baseline performance {round(baseline_score, 4)}.")

    '''
    We consider now four different setups:
    - augment the hardest 5% by 200%
    - augment the hardest 10% by 100%
    - augment the hardest 20% by 50%
    - augment the entire dataset by 10%
    '''
    hard = [5, 10, 20]
    for h in hard:
        print(f"Status: augmenting hardest {h}%...")
        os.mkdir(f'../outputs/results/{dataset_name}/tuning_hardest_{h}')
        X_hard = X_train[ind_knn[:int(len(y_train) * h / 100)]]
        y_hard = y_train[ind_knn[:int(len(y_train) * h / 100)]]
        X_hard_df = pd.DataFrame(X_hard)
        X_hard_df['target'] = y_hard
        ind = 0
        # Infer metadata: categorical and numeric features
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(X_hard_df)
        # Compute average score on all data
        tot_score, _ = train_xgb(X_train, y_train, X_v, y_v, n=10)

        def get_xgboost_score(X_train, y_train):
            # Compute difference in average score with respect to all data
            new_score, _ = train_xgb(X_train, y_train, X_v, y_v, n=n_retrain)
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
            over multiple runs when synthetic data is added to training dataset.
            '''
            nonlocal ind
            mod_class = mods[mod_name]
            params = transform_dict(mod_name, hyperparams)
            mod_instance = mod_class(metadata, 
                                     save_path=f'../outputs/results/{dataset_name}/tuning_hardest_{h}/{ind}.pkl',
                                     epochs=500,
                                     batch_size=len(X_hard_df),
                                     cuda='cuda:1',
                                     patience=50,
                                     weights=df_importance['gain'].values,
                                     **params)
            mod_instance.fit(X_hard_df)
            # Restore best model
            mod_instance = synths[mod_name].load(f'../outputs/results/{dataset_name}/tuning_hardest_{h}/{ind}.pkl')
            ind += 1
            scores = []
            for _ in range(n_retry):
                synthetic_data = mod_instance.sample(len(y_hard))
                X_synth = synthetic_data.drop('target', axis=1).values
                y_synth = np.array(synthetic_data['target'])
                X_train_aug = np.vstack((X_train, X_synth))
                y_train_aug = np.concatenate((y_train, y_synth))
                scores.append(get_xgboost_score(X_train_aug, y_train_aug))
            hyperparams_log = copy.deepcopy(hyperparams)
            hyperparams_log['mod_name'] = mod_name
            append_to_pickle_file(f'../outputs/results/{dataset_name}/hyperparams_hardest_{h}.pkl', hyperparams_log)
            append_to_pickle_file(f'../outputs/results/{dataset_name}/scores_hardest_{h}.pkl', scores)
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
        # Define tuning run
        session = BTBSession(
            tunables=tunables,
            scorer=scoring_function,
            verbose=True
        )
        start_tune = time.time()
        best_prop = session.run(n_tune)
        end_tune = time.time()
        with open(f"../outputs/results/{dataset_name}/notes", "a") as f:
            f.write(f"Seconds to augment hardest {h}%: {end_tune - start_tune}.\n")

    # Lastly we augment the entire dataset by 10%
    print(f"Status: augmenting entire dataset...")
    os.mkdir(f"../outputs/results/{dataset_name}/tuning_tot")
    X_train_df = pd.DataFrame(X_train)
    X_train_df['target'] = y_train
    ind = 0

    def scoring_function(mod_name, hyperparams):
        nonlocal ind
        mod_class = mods[mod_name]
        params = transform_dict(mod_name, hyperparams)
        mod_instance = mod_class(metadata, 
                                 save_path=f'../outputs/results/{dataset_name}/tuning_tot/{ind}.pkl',
                                 epochs=500,
                                 batch_size=len(X_train_df),
                                 cuda='cuda:1',
                                 patience=50,
                                 weights=df_importance['gain'].values,
                                 **params)
        mod_instance.fit(X_train_df)
        # Restore best model
        mod_instance = synths[mod_name].load(f'../outputs/results/{dataset_name}/tuning_tot/{ind}.pkl')
        ind += 1
        scores = []
        for _ in range(n_retry):
            synthetic_data = mod_instance.sample(int(len(y_train) * 0.1))
            X_synth = synthetic_data.drop('target', axis=1).values
            y_synth = np.array(synthetic_data['target'])
            X_train_aug = np.vstack((X_train, X_synth))
            y_train_aug = np.concatenate((y_train, y_synth))
            scores.append(get_xgboost_score(X_train_aug, y_train_aug))
        hyperparams_log = copy.deepcopy(hyperparams)
        hyperparams_log['mod_name'] = mod_name
        append_to_pickle_file(f'../outputs/results/{dataset_name}/hyperparams_tot.pkl', hyperparams_log)
        append_to_pickle_file(f'../outputs/results/{dataset_name}/scores_tot.pkl', scores)
        return np.mean(scores)
    
    session = BTBSession(
        tunables=tunables,
        scorer=scoring_function,
        verbose=True
    )
    start_tune = time.time()
    best_prop = session.run(n_tune)
    end_tune = time.time()
    with open(f"../outputs/results/{dataset_name}/notes", "a") as f:
        f.write(f"Seconds to augment entire dataset: {end_tune - start_tune}.\n")
    return 0

if __name__ == "__main__":
    
    X, y = make_classification(n_samples=10000, 
                               n_features=2, 
                               n_informative=2, 
                               n_redundant=0, 
                               n_clusters_per_class=2, 
                               weights=[0.5, 0.5], 
                               class_sep=1.1,
                               flip_y=0,
                               random_state=42)
    run_targeted_aug(X, y, dataset_name="blobs", n_retrain=5, n_retry=30)

    default_of_credit_card_clients = fetch_ucirepo(id=350)
    X = default_of_credit_card_clients.data.features 
    y = default_of_credit_card_clients.data.targets
    X, y = np.array(X), np.array(y["Y"][X.index])
    run_targeted_aug(X, y, dataset_name="credit_card", n_retrain=5, n_retry=30)
    
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
    X = cdc_diabetes_health_indicators.data.features 
    y = cdc_diabetes_health_indicators.data.targets
    X, y = np.array(X), np.array(y["Diabetes_binary"][X.index])
    run_targeted_aug(X, y, dataset_name="diabetes", n_retrain=5, n_retry=30)
    
    support2 = fetch_ucirepo(id=880) 
    X = support2.data.features 
    y = support2.data.targets
    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            X[column].fillna(-1, inplace=True)
        else:
            X[column].fillna("NULL", inplace=True)
    X, y = np.array(X), np.array(y['death'][X.index])
    run_targeted_aug(X, y, dataset_name="support", needs_encoding=True, n_retrain=5, n_retry=30)
