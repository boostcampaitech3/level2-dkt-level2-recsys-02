import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme(color_codes=True)

def train_valid_split(cv_df, aug_preds=[]):
    valid = cv_df[cv_df['cv_idx'] == True]
    train_cond1 = cv_df['is_valid'] == False
    train_cond2 = cv_df['answerCode'] != -1
    train = cv_df[train_cond1 & train_cond2]
    
    if len(aug_preds) != 0 :
        pseudo_df = cv_df[cv_df['answerCode'] == -1].copy()
        pseudo_df['answerCode'] = aug_preds
        train = pd.concat([train, pseudo_df])
    
    y_train = train['answerCode']
    train = train.drop(['answerCode'], axis=1)

    y_valid = valid['answerCode']
    valid = valid.drop(['answerCode'], axis=1)
    
    return train, y_train, valid, y_valid


def plot_importance(feature_importances, X, output_dir, fig_size = (40, 40)):
    feature_imp = pd.DataFrame({'Value':feature_importances,'Feature':X.columns})
    plt.figure(figsize=fig_size)
    sns.set(font_scale = 3)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 
                                                        ascending=False)[0:len(X.columns)])
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    output_file_path = os.path.join(output_dir, 'lgbm_importances_avg.png')
    plt.savefig(output_file_path)