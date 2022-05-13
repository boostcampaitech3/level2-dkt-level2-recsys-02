def feature_load():
    FEATS = \
    [
        'KnowledgeTag',
             
        'user_correct_answer', 
        'user_total_answer', 
        'user_acc',
        
        'test_mean', 
        'test_sum', 
        'test_std',
             
        'tag_mean',
        'tag_sum', 
        'tag_std', 
        
        'type_mean', 
        'type_sum',
        'type_std',
        
        'question_mean',
        'question_sum',
        'question_std', 
        
        'question_number_mean', 
        'question_number_sum', 
        'question_number_std',
        
        'test_number_mean', 
        'test_number_sum', 
        'test_number_std',
        
        'hour', 

        'accuracy_trend3', 
        'accuracy_trend5', 
        'accuracy_trend10', 
        'accuracy_trend30', 
        'accuracy_trend50', 
        'accuracy_trend100', 
        'accuracy_trend200',
        
        'elapsedTime_v4', 
        
        'user_elapsedTime_mean',
        'user_correct_elapsedTime_mean',
        'user_wrong_elapsedTime_mean',
        
        'question_elapsedTime_mean',
        'question_correct_elapsedTime_mean',
        'question_wrong_elapsedTime_mean',
        
        'past_type_count',
        'past_type_correct',
        'past_type_accuracy',
        'past_type_elapsedTime',
        
        'past_tag_count',
        'past_tag_correct',
        'past_tag_accuracy',
        'past_tag_elapsedTime',
                
        'roll_elapsedTime_mean3',
        'roll_elapsedTime_mean5',
        'roll_elapsedTime_mean10',
        'roll_elapsedTime_mean30',
        'roll_elapsedTime_mean50',
        'roll_elapsedTime_mean100',
        'roll_elapsedTime_mean200',

        'word2vec_wrong_question1',
        'word2vec_wrong_question2',
        'word2vec_wrong_question3',
        'word2vec_wrong_question4',
        'word2vec_wrong_question5',
        'word2vec_wrong_question6',
        'word2vec_wrong_question7',
        'word2vec_wrong_question8',
        'word2vec_wrong_question9',
        'word2vec_wrong_question10',
        
        'svd_question1',
        'svd_question2',
        'svd_question3',
        'svd_question4',
        'svd_question5',
        
        'trueSkill_win_probability',
        
        'correct_trend3', 
        'correct_trend5', 
        'correct_trend10', 
        'correct_trend30', 
        'correct_trend50', 
        'correct_trend100', 
        'correct_trend200',
        
        'accuracy_time_trend10min',
        'accuracy_time_trend10D',
        
        'correct_time_trend10min',
        'correct_time_trend10D',
    ]
    
    print(f'Number of Features: {len(FEATS)}')
    return FEATS