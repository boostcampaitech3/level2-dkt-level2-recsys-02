import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pickle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def probability_of_good_answer(theta, beta, left_asymptote):
    return left_asymptote + (1 - left_asymptote) * sigmoid(theta - beta)


def estimate_probas(test_df, student_parameters, item_parameters, granularity_feature_name='assessmentItemID'):
    probability_of_success_list = []
    
    for student_id, item_id, left_asymptote in tqdm(
        zip(test_df.userID.values, test_df[granularity_feature_name].values, test_df.left_asymptote.values)
    ):
        theta = student_parameters[student_id]['theta'] if student_id in student_parameters else 0
        beta = item_parameters[item_id]['beta'] if item_id in item_parameters else 0

        probability_of_success_list.append(probability_of_good_answer(theta, beta, left_asymptote))

    return probability_of_success_list


def load_agg_data(df):
    test_cond = df['answerCode'] == -1
    valid_cond = df['is_valid'] == True
    agg_df = df[~(test_cond | valid_cond)]
    return agg_df


def update_elapsedTime(df):
    agg_df = load_agg_data(df)
    cond = df['elapsedTime'].isna()
    agg_df = agg_df[agg_df['elapsedTime'].isna() == False]
    
    global_elapsedTime_mean = agg_df['elapsedTime'].mean()
    
    # version 1
    df[f'elapsedTime_v1'] = df['elapsedTime'].values
    df.loc[cond, f'elapsedTime_v1'] = global_elapsedTime_mean
    
    # version 2
    df[f'elapsedTime_v2'] = df['elapsedTime'].values
    question_time_dict = agg_df.groupby('assessmentItemID').elapsedTime.mean().to_dict()
    df.loc[cond, f'elapsedTime_v2'] = df.loc[cond, 'assessmentItemID'].apply(lambda x:question_time_dict[x] if x in question_time_dict.keys() else global_elapsedTime_mean)
    
    # version 3
    df[f'elapsedTime_v3'] = df['elapsedTime'].values
    user_time_dict = agg_df.groupby('userID').elapsedTime.mean().to_dict()
    df.loc[cond, f'elapsedTime_v3'] = df.loc[cond, 'userID'].apply(lambda x:user_time_dict[x] if x in user_time_dict.keys() else global_elapsedTime_mean)

    # version 4        
    df[f'elapsedTime_v4'] = df['elapsedTime'].values
    df.loc[cond, f'elapsedTime_v4'] = df.loc[cond, f'roll_elapsedTime_mean3']
    df['elapsedTime_v4'].fillna(global_elapsedTime_mean, inplace=True)
    
    return df


def update_user_question_elapsedTime(df):
    agg_df = load_agg_data(df)
    
    # 유저 별 all/correct/wrong 걸린 시간 평균
    user_df = agg_df.groupby('userID')[f'elapsedTime_v3'].agg(['mean'])
    user_correct_df = agg_df[agg_df['answerCode'] == 1].groupby('userID')[f'elapsedTime_v3'].agg(['mean'])
    user_wrong_df = agg_df[agg_df['answerCode'] == 0].groupby('userID')[f'elapsedTime_v3'].agg(['mean'])
    
    user_df.columns = [f'user_elapsedTime_mean']
    user_correct_df.columns = [f'user_correct_elapsedTime_mean']
    user_wrong_df.columns = [f'user_wrong_elapsedTime_mean']
    
    # 문제 별 all/correct/wrong 걸린 시간 평균
    question_df = agg_df.groupby('assessmentItemID')[f'elapsedTime_v2'].agg(['mean'])
    question_correct_df = agg_df[agg_df['answerCode'] == 1].groupby('assessmentItemID')[f'elapsedTime_v2'].agg(['mean'])
    question_wrong_df = agg_df[agg_df['answerCode'] == 0].groupby('assessmentItemID')[f'elapsedTime_v2'].agg(['mean'])
    
    question_df.columns = [f'question_elapsedTime_mean']
    question_correct_df.columns = [f'question_correct_elapsedTime_mean']
    question_wrong_df.columns = [f'question_wrong_elapsedTime_mean']
    
    df = pd.merge(df, user_df, on=['userID'], how="left", suffixes=('_del', ''))
    df = pd.merge(df, user_correct_df, on=['userID'], how="left", suffixes=('_del', ''))
    df = pd.merge(df, user_wrong_df, on=['userID'], how="left", suffixes=('_del', ''))
    df = pd.merge(df, question_df, on=['assessmentItemID'], how="left", suffixes=('_del', ''))
    df = pd.merge(df, question_correct_df, on=['assessmentItemID'], how="left", suffixes=('_del', ''))
    df = pd.merge(df, question_wrong_df, on=['assessmentItemID'], how="left", suffixes=('_del', ''))
    
    time_df = df.groupby('userID').elapsedTime.rolling(window=3).sum()
    time_df = time_df.reset_index()[['userID', 'elapsedTime']]
    cond1 = time_df.elapsedTime >= 0
    cond2 = time_df.elapsedTime < 3
    df['randomly_marked'] = cond1 & cond2

    return df


def update_user_correct_ratio(df):
    agg_df = load_agg_data(df)
    corr_df = agg_df.groupby('assessmentItemID')['answerCode'].agg([['corr_ratio', 'mean']]).reset_index()
    corr_df = agg_df[agg_df['answerCode']==0].merge(corr_df, on='assessmentItemID')
    corr_df = corr_df.groupby('userID')['corr_ratio'].agg(['min', 'max', 'mean', 'std']).reset_index()
    corr_df.columns = ['userID', 'corr_min', 'corr_max', 'corr_mean', 'corr_std']
    
    df = pd.merge(df, corr_df, on=['userID'], how='left', suffixes=('_del', ''))
    return df


def update_statistics(df):    
    agg_df = load_agg_data(df)
    
    correct_i = agg_df.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum', 'std'])
    correct_i.columns = ["question_mean", 'question_sum', 'question_std']
    correct_t = agg_df.groupby(['testId'])['answerCode'].agg(['mean', 'sum', 'std'])
    correct_t.columns = ["test_mean", 'test_sum', 'test_std']
    correct_k = agg_df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum', 'std'])
    correct_k.columns = ["tag_mean", 'tag_sum', 'tag_std']
    type_df = agg_df.groupby('testType')['answerCode'].agg(['mean', 'sum', 'std'])
    type_df.columns = ['type_mean', 'type_sum', 'type_std']
    qn_df = agg_df.groupby('questionNumber')['answerCode'].agg(['mean','sum','std'])
    qn_df.columns = ['question_number_mean', 'question_number_sum', 'question_number_std']
    tn_df = agg_df.groupby('testNumber').answerCode.agg(['mean', 'sum', 'std'])
    tn_df.columns = ['test_number_mean', 'test_number_sum', 'test_number_std']
    
    # corr_df = agg_df.groupby('assessmentItemID')['answerCode'].agg([['corr_ratio', 'mean']]).reset_index()
    # corr_df = agg_df[agg_df['answerCode']==0].merge(corr_df, on='assessmentItemID')
    # corr_df = corr_df.groupby('userID')['corr_ratio'].agg(['min', 'max', 'mean', 'std']).reset_index()
    # corr_df.columns = ['userID', 'corr_min', 'corr_max', 'corr_mean', 'corr_std']
    
    df = pd.merge(df, correct_i, on=['assessmentItemID'], how="left", suffixes=('_del', ''))
    df = pd.merge(df, correct_t, on=['testId'], how="left", suffixes=('_del', ''))
    df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left", suffixes=('_del', ''))
    df = pd.merge(df, type_df, on=['testType'], how="left", suffixes=('_del', ''))
    df = pd.merge(df, qn_df, on=['questionNumber'], how='left', suffixes=('_del', ''))
    df = pd.merge(df, tn_df, on=['testNumber'], how='left', suffixes=('_del', ''))
    
    for window_size in [3, 5, 10, 30, 50, 100, 200,]:
        df[f'normalized_accuracy_trend{window_size}'] = df[f'accuracy_trend{window_size}'] - df['question_mean']
        
    for window_size in ['10min','10D']:
        df[f'normalized_accuracy_time_trend{window_size}'] = df[f'accuracy_time_trend{window_size}'] - df['question_mean']
    
    return df


def update_time_slot(df):
    # 날짜, timestamp, 시간
    agg_df = load_agg_data(df)
    
    # 시간대별 정확도, 유저별 공부 시간, 야행성 여부
    hour_dict = agg_df.groupby(['hour'])['answerCode'].mean().to_dict()
    mode_dict = df.groupby(['userID'])['hour'].agg(lambda x: pd.Series.mode(x)[0]).to_dict()
    df['accuracy_per_hour'] = df['hour'].map(hour_dict)
    df['hour_mode'] = df['userID'].map(mode_dict)
    df['is_night'] = (df['hour_mode'] >= 22).values | (df['hour_mode'] < 4).values
    
    # Test 치는데 걸리는 시간
    # df['test_total_time'] = (df['test_end_time'].values - df['test_start_time'].values) / np.timedelta64(1, 's')
    
    return df


def update_dimension_reduction(df, cv_num):
    # Truncated SVD
    SVD_DIM = 5
    with open(f'./assets/cv{cv_num}/svd_question.pickle','rb') as f:
         svd_q_dict = pickle.load(f)
         
    svd_q_df = pd.DataFrame.from_dict(svd_q_dict).T
    cols = [f'svd_question{i+1}' for i in range(SVD_DIM)]
    cols.insert(0, 'assessmentItemID')
    svd_q_df = svd_q_df.reset_index()
    svd_q_df.columns = cols
    df = pd.merge(df, svd_q_df, how='left', on='assessmentItemID')

    with open(f'./assets/cv{cv_num}/svd_user.pickle','rb') as f:
         svd_u_dict = pickle.load(f)
         
    svd_u_df = pd.DataFrame.from_dict(svd_u_dict).T
    cols = [f'svd_user{i+1}' for i in range(SVD_DIM)]
    cols.insert(0, 'userID')
    svd_u_df = svd_u_df.reset_index()
    svd_u_df.columns = cols
    df = pd.merge(df, svd_u_df, how='left', on='userID')
    
    return df


def update_word2vec_embedding(df, cv_num):
    EMB_DIM = 10
    # user's correct question list word2vec 
    with open(f'./assets/cv{cv_num}/word2vec_correct_question.pickle','rb') as f:
         word2vec_correct = pickle.load(f)
    
    emb_correct_df = pd.DataFrame.from_dict(word2vec_correct).T
    cols = [f'word2vec_correct_question{i+1}' for i in range(EMB_DIM)]
    cols.insert(0, 'assessmentItemID')
    emb_correct_df = emb_correct_df.reset_index()
    emb_correct_df.columns = cols
    
    # user's wrong question list word2vec
    with open(f'./assets/cv{cv_num}/word2vec_wrong_question.pickle','rb') as f:
         word2vec_wrong = pickle.load(f)
    
    emb_wrong_df = pd.DataFrame.from_dict(word2vec_wrong).T
    cols = [f'word2vec_wrong_question{i+1}' for i in range(EMB_DIM)]
    cols.insert(0, 'assessmentItemID')
    emb_wrong_df = emb_wrong_df.reset_index()
    emb_wrong_df.columns = cols
    
    df = pd.merge(df, emb_correct_df, how='left', on='assessmentItemID')
    df = pd.merge(df, emb_wrong_df, how='left', on='assessmentItemID')
    
    return df


def update_elo_rating(df, cv_num, win_prob=True):
    with open(f'./assets/cv{cv_num}/elo_student_parameters.pickle','rb') as f:
        student_parameters = pickle.load(f)

    with open(f'./assets/cv{cv_num}/elo_item_parameters.pickle','rb') as f:
        item_parameters = pickle.load(f)
    
    if win_prob :    
        df['left_asymptote']=1/2
        df['elo_win_probability'] = estimate_probas(df, student_parameters, item_parameters)

    student_df = pd.DataFrame.from_dict(student_parameters).T
    student_df.columns = ['elo_theta', 'user_nb_answers']
    student_df['userID'] = student_df.index
    item_df = pd.DataFrame.from_dict(item_parameters).T
    item_df.columns = ['elo_beta', 'item_nb_answers']
    item_df['assessmentItemID'] = item_df.index

    df = pd.merge(df, student_df, how='left', on='userID')
    df = pd.merge(df, item_df, how='left', on='assessmentItemID')

    return df


def update_true_skill(df, cv_num):
    
    df[f'trueSkill_win_probability']= df[f'trueSkill_win_probability_cv{cv_num}'].values
    df[f'trueSkill_user_mu']= df[f'trueSkill_user_mu_cv{cv_num}'].values
    df[f'trueSkill_user_sigma']= df[f'trueSkill_user_sigma_cv{cv_num}'].values
    df[f'trueSkill_question_mu']= df[f'trueSkill_question_mu_cv{cv_num}'].values
    df[f'trueSkill_question_sigma']= df[f'trueSkill_question_sigma_cv{cv_num}'].values
    
    return df


def valid_update(df, cv_num):
    data_dir = '/opt/ml/project/data'
    users_file_path = os.path.join(data_dir, f'cv1_users.pickle')
    with open(users_file_path,'rb') as f:
        users = pickle.load(f)
    train_users = users['train_users']
    test_users = users['test_users']

    valid_cond1 = df['userID'].isin(train_users) == False
    valid_cond2 = df['userID'].isin(test_users) == False
    cv_idx = df[valid_cond1&valid_cond2].groupby('userID', as_index=False).nth(-cv_num).index
    valid_idx = df[valid_cond1&valid_cond2].groupby('userID').tail(cv_num).index
    
    df['cv_idx'] = False
    df['is_valid'] = False
    
    df.loc[cv_idx, 'cv_idx'] = True
    df.loc[valid_idx, 'is_valid'] = True
    
    return df


def state_update(data, cv_num):
    data = valid_update(data, cv_num)
    data = update_elapsedTime(data)
    data = update_user_question_elapsedTime(data)
    data = update_statistics(data)
    data = update_dimension_reduction(data, cv_num)
    data = update_word2vec_embedding(data, cv_num)
    data = update_true_skill(data, cv_num)
    
    data.fillna(0, inplace=True)

    return data[[c for c in data.columns if not c.endswith('_del')]]