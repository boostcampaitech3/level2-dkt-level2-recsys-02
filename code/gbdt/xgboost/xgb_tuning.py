import pandas as pd
import os
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import pickle
import optuna
from fe_add import feature_engineering
from fe_update import state_update
from features import feature_load
from utils import train_valid_split


def objective(params, train, y_train, valid, y_valid):    
    lgb_train = lgb.Dataset(train[FEATS], y_train)
    lgb_valid = lgb.Dataset(valid[FEATS], y_valid)

    model = lgb.train(
        params, 
        lgb_train, 
        valid_sets=[lgb_train, lgb_valid], 
        verbose_eval=0, 
        num_boost_round=2000, 
        early_stopping_rounds=100,
    )
    val_pred = model.predict(valid[FEATS])
    score = roc_auc_score(y_valid, val_pred)
    return score


def objective_cv(trial):
    params = \
    {
        'objective': 'binary',
        'metric': 'auc',
        'feature_pre_filter': False,
        'num_leaves': trial.suggest_int('num_leaves', 32, 512),
        'max_depth': trial.suggest_int('max_depth', 4, 16),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 16),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 4, 80),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 1.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 1.0),
    }
    
    score_list = []
    for cv_num in range(1, 6):
        cv_df = state_update(fe_df, cv_num)
        
        train, y_train, valid, y_valid = train_valid_split(cv_df, aug_preds)
    
        score = objective(params, train, y_train, valid, y_valid)
        score_list.append(score)
    
    return np.mean(score_list)


if __name__ == '__main__':
    ## DATA 로드
    data_dir = '/opt/ml/project/data/'
    csv_file_path = os.path.join(data_dir, 'total_data_v2.csv')
    df = pd.read_csv(csv_file_path, parse_dates=['Timestamp'])
    df = df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
    
    pred_dir = './output'
    pred_path = os.path.join(pred_dir, 'exp3.csv')
    pred_df = pd.read_csv(pred_path)
    aug_preds = np.where(pred_df['prediction'] >= 0.5, 1, 0)
    
    fe_df = feature_engineering(df)
    FEATS = feature_load()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_cv, n_trials=20)
    
    with open('./assets/kfold_params.pickle', 'wb') as fbp:
        pickle.dump(study.best_trial.params, fbp)