import os

import pickle
import pandas as pd
import torch


def prepare_dataset(device, basepath, verbose=True, logger=None):
    data = load_data(basepath)
    train_data, valid_data, test_data = separate_data(basepath, data)
    id2index = indexing_data(data)
    train_data_proc = process_data(train_data, id2index, device)
    valid_data_proc = process_data(valid_data, id2index, device)
    test_data_proc = process_data(test_data, id2index, device)

    if verbose:
        print_data_stat(train_data, "Train", logger=logger)
        print_data_stat(valid_data, "Valid", logger=logger)
        print_data_stat(test_data, "Test", logger=logger)

    return train_data_proc, valid_data_proc, test_data_proc, len(id2index)


def load_data(basepath):
    path = os.path.join(basepath, "total_data.csv")
    data = pd.read_csv(path)
    data.drop_duplicates(
        subset=["userID", "assessmentItemID"], keep="last", inplace=True
    )
    data = data.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)

    return data


def separate_data(basepath, data):
    users_file_path = os.path.join(basepath, 'cv1_users.pickle')
    with open(users_file_path,'rb') as f:
        users = pickle.load(f)
    train_users, test_users = users['train_users'], users['test_users']
    
    test_cond = data['answerCode'] == -1
    valid_cond1 = data['userID'].isin(test_users) == False
    valid_cond2 = data['userID'].isin(train_users) == False
    valid_cond3 = data['userID'] != data['userID'].shift(-1)

    train_data = data[~test_cond & ~(valid_cond1 & valid_cond2 & valid_cond3)].copy()
    valid_data = data[valid_cond1 & valid_cond2 & valid_cond3].copy()
    test_data = data[test_cond].copy()

    return train_data, valid_data, test_data


def indexing_data(data):
    userid, itemid = (
        sorted(list(set(data.userID))),
        sorted(list(set(data.assessmentItemID))),
    )
    n_user, n_item = len(userid), len(itemid)

    userid_2_index = {v: i for i, v in enumerate(userid)}
    itemid_2_index = {v: i + n_user for i, v in enumerate(itemid)}
    id_2_index = dict(userid_2_index, **itemid_2_index)

    return id_2_index


def process_data(data, id_2_index, device):
    edge, label = [], []
    for user, item, acode in zip(data.userID, data.assessmentItemID, data.answerCode):
        uid, iid = id_2_index[user], id_2_index[item]
        edge.append([uid, iid])
        label.append(acode)

    edge = torch.LongTensor(edge).T
    label = torch.LongTensor(label)

    return dict(edge=edge.to(device), label=label.to(device))


def print_data_stat(data, name, logger):
    userid, itemid = list(set(data.userID)), list(set(data.assessmentItemID))
    n_user, n_item = len(userid), len(itemid)

    logger.info(f"{name} Dataset Info")
    logger.info(f" * Num. Users    : {n_user}")
    logger.info(f" * Max. UserID   : {max(userid)}")
    logger.info(f" * Num. Items    : {n_item}")
    logger.info(f" * Num. Records  : {len(data)}")
