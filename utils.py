
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
import os
import pandas as pd


def random_split(n=11988, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Use 9:1:1 split"""
    p_train = train_ratio
    p_val = val_ratio
    p_test = test_ratio

    n = 11988  # original 12000 patients, remove 12 outliers
    n_train = round(n * p_train)
    n_val = round(n * p_val)
    n_test = n - (n_train + n_val)
    p = np.random.permutation(n)
    idx_train = p[:n_train]
    idx_val = p[n_train:n_train + n_val]
    idx_test = p[n_train + n_val:]
    return idx_train, idx_val, idx_test


def get_data_split(base_path, split_path, split_type='random', reverse=False, baseline=True, dataset='P12'):
    # load data
    if dataset == 'P12':
        Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
        dataset_prefix = ''
    elif dataset == 'P19':
        Pdict_list = np.load(base_path + '/processed_data/PT_dict_list_6.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes_6.npy', allow_pickle=True)
        dataset_prefix = 'P19_'
    elif dataset == 'eICU':
        Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
        dataset_prefix = 'eICU_'
    elif dataset == 'PAM':
        Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
        dataset_prefix = ''  # not applicable

    show_statistics = False
    if show_statistics:
        idx_under_65 = []
        idx_over_65 = []

        idx_male = []
        idx_female = []

        # variables for statistics
        all_ages = []
        female_count = 0
        male_count = 0
        all_BMI = []

        X_static = np.zeros((len(Pdict_list), len(Pdict_list[0]['extended_static'])))
        for i in range(len(Pdict_list)):
            X_static[i] = Pdict_list[i]['extended_static']
            age, gender_0, gender_1, height, _, _, _, _, weight = X_static[i]
            if age > 0:
                all_ages.append(age)
                if age < 65:
                    idx_under_65.append(i)
                else:
                    idx_over_65.append(i)
            if gender_0 == 1:
                female_count += 1
                idx_female.append(i)
            if gender_1 == 1:
                male_count += 1
                idx_male.append(i)
            if height > 0 and weight > 0:
                all_BMI.append(weight / ((height / 100) ** 2))

    if baseline==True:
        BL_path = ''
    else:
        BL_path = 'baselines/'

    if split_type == 'random':
        # load random indices from a split
        idx_train, idx_val, idx_test = np.load(base_path + split_path, allow_pickle=True)
    elif split_type == 'age':
        if reverse == False:
            idx_train = np.load(BL_path+'saved/' + dataset_prefix + 'idx_under_65.npy', allow_pickle=True)
            idx_vt = np.load(BL_path+'saved/' + dataset_prefix + 'idx_over_65.npy', allow_pickle=True)
        elif reverse == True:
            idx_train = np.load(BL_path+'saved/' + dataset_prefix + 'idx_over_65.npy', allow_pickle=True)
            idx_vt = np.load(BL_path+'saved/' + dataset_prefix + 'idx_under_65.npy', allow_pickle=True)

        np.random.shuffle(idx_vt)
        idx_val = idx_vt[:round(len(idx_vt) / 2)]
        idx_test = idx_vt[round(len(idx_vt) / 2):]
    elif split_type == 'gender':
        if reverse == False:
            idx_train = np.load(BL_path+'saved/' + dataset_prefix + 'idx_male.npy', allow_pickle=True)
            idx_vt = np.load(BL_path+'saved/' + dataset_prefix + 'idx_female.npy', allow_pickle=True)
        elif reverse == True:
            idx_train = np.load(BL_path+'saved/' + dataset_prefix + 'idx_female.npy', allow_pickle=True)
            idx_vt = np.load(BL_path+'saved/' + dataset_prefix + 'idx_male.npy', allow_pickle=True)

        np.random.shuffle(idx_vt)
        idx_val = idx_vt[:round(len(idx_vt) / 2)]
        idx_test = idx_vt[round(len(idx_vt) / 2):]

    # extract train/val/test examples
    Ptrain = Pdict_list[idx_train]
    Pval = Pdict_list[idx_val]
    Ptest = Pdict_list[idx_test]

    y = arr_outcomes[:, -1].reshape((-1, 1))
    ytrain = y[idx_train]
    yval = y[idx_val]
    ytest = y[idx_test]

    return Ptrain, Pval, Ptest, ytrain, yval, ytest


def getStats(P_tensor):
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    mf = np.zeros((F, 1))
    stdf = np.ones((F, 1))
    eps = np.array(1e-7)
    for f in range(F):
        vals_f = Pf[f, :]
        vals_f = vals_f[vals_f > 0]
        mf[f] = np.mean(vals_f)
        stdf[f] = np.std(vals_f)
        stdf[f] = np.maximum(stdf[f], eps)
    return mf, stdf


def mask_normalize(P_tensor, mf, stdf):
    """ Normalize time series variables. Missing ones are set to zero after normalization. """
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    M = 1*(P_tensor > 0) + 0*(P_tensor <= 0)
    M_3D = M.transpose((2, 0, 1)).reshape(F, -1)
    for f in range(F):
        Pf[f] = (Pf[f]-mf[f])/(stdf[f]+1e-18)
    Pf = Pf * M_3D
    Pnorm_tensor = Pf.reshape((F, N, T)).transpose((1, 2, 0))
    Pfinal_tensor = np.concatenate([Pnorm_tensor, M], axis=2)  # ????? why concat with mask: when it pass in model forward, it will be used to mask some features
    return Pfinal_tensor


def getStats_static(P_tensor, dataset='P12'):
    N, S = P_tensor.shape
    Ps = P_tensor.transpose((1, 0))
    ms = np.zeros((S, 1))
    ss = np.ones((S, 1))

    if dataset == 'P12':
        # ['Age' 'Gender=0' 'Gender=1' 'Height' 'ICUType=1' 'ICUType=2' 'ICUType=3' 'ICUType=4' 'Weight']
        bool_categorical = [0, 1, 1, 0, 1, 1, 1, 1, 0]
    elif dataset == 'P19':
        # ['Age' 'Gender' 'Unit1' 'Unit2' 'HospAdmTime' 'ICULOS']
        bool_categorical = [0, 1, 0, 0, 0, 0]
    elif dataset == 'eICU':
        # ['apacheadmissiondx' 'ethnicity' 'gender' 'admissionheight' 'admissionweight'] -> 399 dimensions
        bool_categorical = [1] * 397 + [0] * 2

    for s in range(S):
        if bool_categorical == 0:  # if not categorical
            vals_s = Ps[s, :]
            vals_s = vals_s[vals_s > 0]
            ms[s] = np.mean(vals_s)
            ss[s] = np.std(vals_s)
    return ms, ss


def mask_normalize_static(P_tensor, ms, ss):
    N, S = P_tensor.shape
    Ps = P_tensor.transpose((1, 0))

    # input normalization
    for s in range(S):
        Ps[s] = (Ps[s] - ms[s]) / (ss[s] + 1e-18)

    # set missing values to zero after normalization
    for s in range(S):
        idx_missing = np.where(Ps[s, :] <= 0)
        Ps[s, idx_missing] = 0

    # reshape back
    Pnorm_tensor = Ps.reshape((S, N)).transpose((1, 0))
    return Pnorm_tensor


def tensorize_normalize(P, y, mf, stdf, ms, ss):
    T, F = P[0]['arr'].shape
    D = len(P[0]['extended_static'])

    P_tensor = np.zeros((len(P), T, F))
    P_time = np.zeros((len(P), T, 1))
    P_static_tensor = np.zeros((len(P), D))
    P_length = np.zeros(len(P))
    for i in range(len(P)):
        P_tensor[i] = P[i]['arr']
        P_time[i] = P[i]['time']
        P_static_tensor[i] = P[i]['extended_static']
        P_length[i] = P[i]['length']
    P_tensor = mask_normalize(P_tensor, mf, stdf)
    P_tensor = torch.Tensor(P_tensor)

    # P_time = torch.Tensor(P_time) / 60.0  # convert mins to hours
    P_time = torch.Tensor(P_time) # convert mins to hours
    P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
    P_static_tensor = torch.Tensor(P_static_tensor)

    y_tensor = y
    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
    return P_tensor, P_static_tensor, P_time, y_tensor, P_length


def tensorize_normalize_other(P, y, mf, stdf):
    T, F = P[0].shape
    P_time = np.zeros((len(P), T, 1))
    P_length = np.zeros(len(P))
    for i in range(len(P)):
        tim = torch.linspace(0, T, T).reshape(-1, 1)
        P_time[i] = tim
        P_length[i] = len(P[i])
        
        
    P_tensor = mask_normalize(P, mf, stdf)
    P_tensor = torch.Tensor(P_tensor)

    P_time = torch.Tensor(P_time) / 60.0

    y_tensor = y
    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
    return P_tensor, None, P_time, y_tensor, P_length


def masked_softmax(A, epsilon=0.000000001):
    A_max = torch.max(A, dim=1, keepdim=True)[0]
    A_exp = torch.exp(A - A_max)
    A_exp = A_exp * (A != 0).float()
    A_softmax = A_exp / (torch.sum(A_exp, dim=0, keepdim=True) + epsilon)
    return A_softmax


def random_sample(idx_0, idx_1, B, replace=False):
    """ Returns a balanced sample of tensors by randomly sampling without replacement. """
    idx0_batch = np.random.choice(idx_0, size=int(B / 2), replace=replace)
    idx1_batch = np.random.choice(idx_1, size=int(B / 2), replace=replace)
    idx = np.concatenate([idx0_batch, idx1_batch], axis=0)
    return idx

def sample_data(args, Pval_tensor, Ptest_tensor):
    num_all_features =int(Pval_tensor.shape[2] / 2)
    num_missing_features = round(args.missingratio * num_all_features)
    for i, patient in enumerate(Pval_tensor):
        idx = np.random.choice(num_all_features, num_missing_features, replace=False)
        patient[:, idx] = torch.zeros(Pval_tensor.shape[1], num_missing_features)
        Pval_tensor[i] = patient
    for i, patient in enumerate(Ptest_tensor):
        idx = np.random.choice(num_all_features, num_missing_features, replace=False)
        patient[:, idx] = torch.zeros(Ptest_tensor.shape[1], num_missing_features)
        Ptest_tensor[i] = patient


def evaluate_metrics(args, y_true, y_pred_score, y_pred, mode='Training', n_classes=2):
    base_path = './results'
    filename = f'{mode}_results.csv'
    path = os.path.join(base_path, filename)
    if n_classes != 2:
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_score, average='macro', multi_class='ovr')
        aupr = average_precision_score(y_true, y_pred_score, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        pr_score = precision_score(y_true, y_pred, average='macro')
        re_score = recall_score(y_true, y_pred, average='macro')
    else:
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_score[:, 1])
        aupr = average_precision_score(y_true, y_pred_score[:, 1])
        f1 = f1_score(y_true, y_pred)
        pr_score = precision_score(y_true, y_pred)
        re_score = recall_score(y_true, y_pred)
    
    new_result = {
        'auc': [round(auc, 4)],
        'aupr': [round(aupr, 4)],
        'acc': [round(acc, 4)],
        'f1': [round(f1, 4)],
        'precision': [round(pr_score, 4)],
        'recall': [round(re_score, 4)]
    }
    new_result = pd.DataFrame(new_result)

    if os.path.exists(path):
        df = pd.read_csv(path)
        new_result = pd.concat([df, new_result], ignore_index=True)
        
    new_result.to_csv(path, index=False)
    return new_result