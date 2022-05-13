import numpy as np

def add_roll_elapsdTime(df):
    df['estimated_elapsedTime'] = df['elapsedTime'].isna()
    
    for window_size in [3, 5, 10, 30, 50, 100, 200]:
        df[f'roll_elapsedTime_mean{window_size}'] = \
            df.groupby(['userID'])[f'elapsedTime'].rolling(window_size, min_periods=1).mean().values
    
        df[f'roll_elapsedTime_mean{window_size}'].fillna(0, inplace=True)
        
    return df


def add_user_accuracy(df):
    df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df['user_correct_answer'].fillna(0, inplace=True)
    df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
    df['user_acc'] = df['user_correct_answer']/df['user_total_answer']
    df['user_acc'].fillna(0, inplace=True)
    return df


def add_accuracy_trend(df):
    for window_size in [3, 5, 10, 30, 50, 100, 200]:
        user_df = df.groupby(df['userID']).shift(1)
        accuracy_trend = user_df.groupby(df['userID']).answerCode.rolling(window=window_size, min_periods=1).mean()
        correct_trend = user_df.groupby(df['userID']).answerCode.rolling(window=window_size, min_periods=1).sum()
        
        df[f'accuracy_trend{window_size}'] = accuracy_trend.values
        df[f'accuracy_trend{window_size}'].fillna(0, inplace=True)
        # df[f'normalized_accuracy_trend{window_size}'] = df[f'accuracy_trend{window_size}'] - df['question_mean']
        df[f'correct_trend{window_size}'] = correct_trend.values
        df[f'correct_trend{window_size}'].fillna(0, inplace=True)
    
    df['shift'] = df.groupby('userID').answerCode.shift(1)
    
    for window_size in ['10min','1h','10h','1D','10D']:
        temp_accuracy_arr = np.zeros(len(df))
        temp_correct_arr = np.zeros(len(df))

        for user_id, temp_df in df.groupby('userID'):
            idx = temp_df.index
            accuracy_time_trend = temp_df.set_index('Timestamp')['shift'].rolling(window_size, min_periods=1).mean()
            correct_time_trend = temp_df.set_index('Timestamp')['shift'].rolling(window_size, min_periods=1).sum()
            temp_accuracy_arr[idx] = accuracy_time_trend
            temp_correct_arr[idx] = correct_time_trend

        df[f'accuracy_time_trend{window_size}'] = temp_accuracy_arr
        df[f'accuracy_time_trend{window_size}'].fillna(0, inplace=True)
        # df[f'normalized_accuracy_time_trend{window_size}'] = temp_accuracy_arr - df['question_mean'].values
        df[f'correct_time_trend{window_size}'] = temp_correct_arr
        df[f'correct_time_trend{window_size}'].fillna(0, inplace=True)
    
    return df


def add_accuracy_on_past_attempts(df):
    # 과거 똑같은 문제 count/correct/accuracy
    df['past_question_count'] = df.groupby(['userID', 'assessmentItemID']).cumcount()
    df['shift'] = df.groupby(['userID', 'assessmentItemID'])['answerCode'].shift().fillna(0)
    df['past_question_correct'] = df.groupby(['userID', 'assessmentItemID'])['shift'].cumsum()
    df['past_question_accuracy'] = (df['past_question_correct'] / df['past_question_count']).fillna(0)
    
    # 과거 똑같은 태그 count/correct/accuracy
    df['past_tag_count'] = df.groupby(['userID', 'KnowledgeTag']).cumcount()
    df['shift'] = df.groupby(['userID', 'KnowledgeTag'])['answerCode'].shift().fillna(0)
    df['past_tag_correct'] = df.groupby(['userID', 'KnowledgeTag'])['shift'].cumsum()
    df['past_tag_accuracy'] = (df['past_tag_correct'] / df['past_tag_count']).fillna(0)
    
    # 과거 똑같은 Type count/correct/accuracy
    df['past_type_count'] = df.groupby(['userID', 'testType']).cumcount()
    df['shift'] = df.groupby(['userID', 'testType'])['answerCode'].shift().fillna(0)
    df['past_type_correct'] = df.groupby(['userID', 'testType'])['shift'].cumsum()
    df['past_type_accuracy'] = (df['past_type_correct'] / df['past_type_count']).fillna(0)
    
    # 과거 똑같은 문제 푼지 얼마나 되었는지
    shift = df.groupby(['userID', 'assessmentItemID'])['Timestamp'].shift()
    last_question_elapsedTime = (df['Timestamp'].values - shift.values) / np.timedelta64(1, 's')
    last_question_elapsedTime[np.isnan(last_question_elapsedTime)] = 0
    df['past_question_elapsedTime'] = last_question_elapsedTime

    # 과거 똑같은 태그 푼지 얼마나 되었는지
    shift = df.groupby(['userID', 'KnowledgeTag'])['Timestamp'].shift()
    last_tag_elapsedTime = (df['Timestamp'].values - shift.values) / np.timedelta64(1, 's')
    last_tag_elapsedTime[np.isnan(last_tag_elapsedTime)] = 0
    df['past_tag_elapsedTime'] = last_tag_elapsedTime

    # 과거 똑같은 Type 푼지 얼마나 되었는지
    shift = df.groupby(['userID', 'testType'])['Timestamp'].shift()
    last_tag_elapsedTime = (df['Timestamp'].values - shift.values) / np.timedelta64(1, 's')
    last_tag_elapsedTime[np.isnan(last_tag_elapsedTime)] = 0
    df['past_type_elapsedTime'] = last_tag_elapsedTime
    
    return df


def add_time(df):
    df['day'] = df.Timestamp.dt.day
    df['time'] = df.Timestamp.apply(lambda x: x.value // 10**9)
    df['hour'] = df['Timestamp'].transform(lambda x: x.dt.hour)
    
    return df
    

def feature_engineering(df):
    df = add_time(df)
    df = add_roll_elapsdTime(df)
    df = add_user_accuracy(df)
    df = add_accuracy_trend(df)
    df = add_accuracy_on_past_attempts(df)
    
    return df