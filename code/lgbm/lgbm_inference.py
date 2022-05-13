import pandas as pd
import os
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

from fe_add import feature_engineering
from fe_update import state_update
from features import feature_load
from utils import plot_importance, train_valid_split


if __name__ == '__main__':
    ## DATA 로드
    data_dir = '/opt/ml/project/data/'
    csv_file_path = os.path.join(data_dir, 'total_data_v2.csv')
    df = pd.read_csv(csv_file_path, parse_dates=['Timestamp'])
    df = df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)

    fe_df = feature_engineering(df)    
    FEATS = feature_load()
    
    total_preds = np.zeros(len(df[df['answerCode'] == -1]))
    feature_importances  = np.zeros(len(FEATS))
    auc_list = []
    acc_list = []
    cv_len = 5
    
    pred_dir = '/opt/ml/project/notebooks/output'
    pred_path = os.path.join(pred_dir, 'optuna20.csv')
    pred_df = pd.read_csv(pred_path)
    aug_preds = np.where(pred_df['prediction'] >= 0.5, 1, 0)
    
    with open('./assets/kfold_params.pickle', 'rb') as fbp:
        best_params = pickle.load(fbp)

    for cv_num in range(1, 1+cv_len):
        cv_df = state_update(fe_df, cv_num)
        
        test_df = cv_df[cv_df['answerCode'] == -1]
        test_users = test_df.userID.unique()
        test_df = test_df.drop(['answerCode'], axis=1)
        
        train, y_train, valid, y_valid = train_valid_split(cv_df, aug_preds)

        lgb_train = lgb.Dataset(train[FEATS], y_train)
        lgb_valid = lgb.Dataset(valid[FEATS], y_valid)
        
        model = lgb.train(
            {'objective':'binary', **best_params},
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            verbose_eval=0,
            num_boost_round=2000,
            early_stopping_rounds=100,
        )
        
        feature_importances += model.feature_importance()
        
        preds = model.predict(valid[FEATS])
        acc = accuracy_score(y_valid, np.where(preds >= 0.5, 1, 0))
        auc = roc_auc_score(y_valid, preds)

        print(f'[{cv_num}] VALID AUC : {auc} ACC : {acc}\n')
        
        auc_list.append(auc)
        acc_list.append(acc)
        
        total_preds += model.predict(test_df[FEATS]) / cv_len
    
    print(f'AUC : {np.mean(auc_list):.4f} ACC : {np.mean(acc_list):.4f}\n')
    
    # SAVE OUTPUT
    output_dir = 'output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_importance(feature_importances, train[FEATS], output_dir)
    
    write_path = os.path.join(output_dir, "exp3.csv")
    with open(write_path, 'w', encoding='utf8') as w:   
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id,p))