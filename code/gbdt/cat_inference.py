import pandas as pd
import os
import numpy as np
import catboost as cb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

from fe_add import feature_engineering
from fe_update import state_update
from features import feature_load
from utils import train_valid_split


if __name__ == '__main__':
    ## DATA 로드
    data_dir = '/opt/ml/project/data/'
    csv_file_path = os.path.join(data_dir, 'total_data_v2.csv')
    df = pd.read_csv(csv_file_path, parse_dates=['Timestamp'])
    df = df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)

    fe_df = feature_engineering(df)    
    FEATS = feature_load()
    
    total_preds = np.zeros(len(df[df['answerCode'] == -1]))
    auc_list = []
    acc_list = []
    cv_len = 5
    
    pred_dir = 'output/'
    pred_path = os.path.join(pred_dir, 'exp3.csv')
    pred_df = pd.read_csv(pred_path)
    aug_preds = np.where(pred_df['prediction'] >= 0.5, 1, 0)
    
    # with open('./assets/cat_params.pickle', 'rb') as fbp:
    #     best_params = pickle.load(fbp)

    for cv_num in range(1, 1+cv_len):
        cv_df = state_update(fe_df, cv_num)
        
        test_df = cv_df[cv_df['answerCode'] == -1]
        test_users = test_df.userID.unique()
        test_df = test_df.drop(['answerCode'], axis=1)
        
        valid = cv_df[cv_df['cv_idx'] == True]
        train_cond1 = cv_df['is_valid'] == False
        train_cond2 = cv_df['answerCode'] != -1
        train = cv_df[train_cond1 & train_cond2]

        pseudo_df = cv_df[cv_df['answerCode'] == -1].copy()
        pseudo_df['answerCode'] = aug_preds
        train = pd.concat([train, pseudo_df])

        # X, y 값 분리
        y_train = train['answerCode']
        train = train.drop(['answerCode'], axis=1)

        y_valid = valid['answerCode']
        valid = valid.drop(['answerCode'], axis=1)

        model = CatBoostClassifier(num_boost_round=2000)

        model.fit(
            train[FEATS], 
            y_train, 
            eval_set=[(valid[FEATS], y_valid)], 
            verbose=100, 
            early_stopping_rounds=100,
        )
        
        preds = model.predict_proba(valid[FEATS])[:, 1]
        acc = accuracy_score(y_valid, np.where(preds >= 0.5, 1, 0))
        auc = roc_auc_score(y_valid, preds)

        print(f'[{cv_num}] VALID AUC : {auc} ACC : {acc}\n')
        
        auc_list.append(auc)
        acc_list.append(acc)
        
        total_preds += model.predict_proba(test_df[FEATS])[:, 1] / cv_len
    
    print(f'AUC : {np.mean(auc_list):.4f} ACC : {np.mean(acc_list):.4f}\n')
    
    # SAVE OUTPUT
    output_dir = 'output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    write_path = os.path.join(output_dir, "cat_exp1.csv")
    with open(write_path, 'w', encoding='utf8') as w:   
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id,p))