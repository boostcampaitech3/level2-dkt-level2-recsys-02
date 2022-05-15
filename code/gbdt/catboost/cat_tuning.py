import pandas as pd
import os
import numpy as np
import catboost as cb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import pickle
import optuna
from fe_add import feature_engineering
from fe_update import state_update
from features import feature_load
from utils import train_valid_split


def objective(params, train, y_train, valid, y_valid):    
    model = CatBoostClassifier(**params)

    model.fit(
        train[FEATS], 
        y_train, 
        eval_set=[(valid[FEATS], y_valid)], 
        verbose=0, 
        early_stopping_rounds=30,
    )
    
    val_pred = model.predict_proba(valid[FEATS])[:, 1]
    score = roc_auc_score(y_valid, val_pred)
    return score


def objective_cv(trial):
    params = \
    {
        'num_boost_round':1000,
        "max_depth":trial.suggest_int("max_depth", 4, 16),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        'bagging_temperature': trial.suggest_float("bagging_temperature", 0, 10),
        "l2_leaf_reg":trial.suggest_float("l2_leaf_reg",1e-8,1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 4, 80),
        "max_bin": trial.suggest_int("max_bin", 200, 500),
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
    
    fe_df = feature_engineering(df)
    FEATS = feature_load()
    
    pred_dir = 'output/'
    pred_path = os.path.join(pred_dir, 'exp3.csv')
    pred_df = pd.read_csv(pred_path)
    aug_preds = np.where(pred_df['prediction'] >= 0.5, 1, 0)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_cv, n_trials=10)
    
    with open('./assets/cat_params.pickle', 'wb') as fbp:
        pickle.dump(study.best_trial.params, fbp)