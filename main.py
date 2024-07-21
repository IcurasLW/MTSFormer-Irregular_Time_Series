import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, average_precision_score, precision_score, recall_score, f1_score
# from models_rd import *
from Irregular_Time_Series.MTSFormer.utils import *
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm, trange
import warnings
import argparse
# from torch_geometric.data import Data
from models_V4 import Transformer_Modi
from sklearn.metrics import roc_auc_score, accuracy_score
from losses import FocalLoss    
import scipy.sparse as sp
import pandas as pd

warnings.filterwarnings("ignore")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='P19', choices=['P12', 'P19', 'PAM']) #
parser.add_argument('--missingratio', type=float, default=0) #
parser.add_argument('--splittype', type=str, default='random', choices=['random', 'age', 'gender'], help='only use for P12 and P19')
parser.add_argument('--reverse', default=False, help='if True,use female, older for tarining; if False, use female or younger for training') #
parser.add_argument('--feature_removal_level', type=str, default='no_remove', choices=['set', 'sample'],
                    help='use this only when splittype==random; otherwise, set as no_removal') #
parser.add_argument('--predictive_label', type=str, default='mortality', choices=['mortality'],
                    help='use this only with P12 dataset (mortality or length of stay)')
args, unknown = parser.parse_known_args()


if args.dataset == 'P12':
    base_path = './P12data'
elif args.dataset == 'P19':
    base_path = './P19data'
elif args.dataset == 'PAM':
    base_path = './PAMdata'


if args.dataset == 'P12':
    d_static = 9
    d_inp = 36
    d_hid = 300
    n_head = 2
    n_layers = 1
    static_info = 1
    max_len = 215
    n_classes = 2
    batch_size = 128
    save_critiria = 'auc'
    dropout=0.25

elif args.dataset == 'P19':
    d_static = 6
    d_inp = 34
    d_hid = 300
    n_head = 2
    n_layers = 1
    static_info = 1
    max_len = 60
    n_classes = 2
    batch_size = 128
    save_critiria = 'auc'
    dropout=0.25

elif args.dataset == 'PAM':
    d_static = 0
    d_inp = 17
    d_hid = 300
    n_head = 4
    n_layers = 4
    static_info = None
    max_len = 600
    n_classes = 8
    batch_size = 64
    save_critiria = 'f1'
    dropout=0.2

n_splits = 5
subset = False
baseline = False  # always False for Raindrop
split = args.splittype  # possible values: 'random', 'age', 'gender'
reverse = args.reverse  # False or True
feature_removal_level = args.feature_removal_level  # 'set', 'sample'


def Multiclass_training(args, model, loss_fn, optimizer, data, length_tensor, time_tensor, y_true, batch_size, n_classes, split_id):
    
    n_batches = (data.shape[1] // batch_size) + 1
    losses = []
    y_pred = []
    y_pred_score = []
    y_true = []
    model.train()
    for b_i in trange(n_batches):
        optimizer.zero_grad()
        out = model.forward(data[:, b_i*batch_size:(b_i+1)*batch_size, :], 
                                time = time_tensor[b_i*batch_size:(b_i+1)*batch_size], 
                                seq_len = length_tensor[b_i*batch_size:(b_i+1)*batch_size])
        
        y_true_i = ytrain_tensor[b_i*batch_size:(b_i+1)*batch_size]
        loss = loss_fn(out, y_true_i)
        out = torch.nn.functional.softmax(out, dim=1)
        loss.backward()
        optimizer.step()
        
        _, y_pred_batch = torch.max(out, dim=1)
        y_pred_score.extend(out.detach().cpu().numpy())
        y_pred.extend(y_pred_batch.detach().cpu().numpy())
        y_true.extend(y_true_i.detach().cpu().numpy())
        losses.append(loss.item())
        
    y_pred_score = np.stack(y_pred_score)
    evaluate_metrics(args, y_true, y_pred_score, y_pred, mode=f'Training_split_{split_id}', n_classes=n_classes)




def Binary_training(args, model, loss_fn, optimizer, data, length_tensor, time_tensor, y_true, batch_size, n_classes, split_id):
    
    # upsample training batch
    idx_0 = np.where(ytrain == 0)[0]
    idx_1 = np.where(ytrain == 1)[0]
    n0, n1 = len(idx_0), len(idx_1)
    expanded_idx_1 = np.concatenate([idx_1, idx_1, idx_1], axis=0)
    expanded_n1 = len(expanded_idx_1)
    K0 = n0 // int(batch_size / 2)
    K1 = expanded_n1 // int(batch_size / 2)
    n_batches = np.min([K0, K1])
    np.random.shuffle(expanded_idx_1)
    I1 = expanded_idx_1
    np.random.shuffle(idx_0)
    I0 = idx_0
    
    losses = []
    y_pred = []
    y_pred_score = []
    y_true = []
    model.train()
    for b_i in trange(n_batches):
        idx0_batch = I0[b_i * int(batch_size / 2):(b_i + 1) * int(batch_size / 2)] # 64
        idx1_batch = I1[b_i * int(batch_size / 2):(b_i + 1) * int(batch_size / 2)] # 64
        idx = np.concatenate([idx0_batch, idx1_batch], axis=0)
        np.random.shuffle(idx)
        optimizer.zero_grad()
        out = model.forward(data[:, idx, :], 
                            time = time_tensor[idx], 
                            seq_len=length_tensor[idx])
        
        y_true_i = ytrain_tensor[idx]
        loss = loss_fn(out, y_true_i)
        out = torch.nn.functional.softmax(out, dim=1)
        loss.backward()
        optimizer.step()
        
        _, y_pred_batch = torch.max(out, dim=1)
        y_pred_score.extend(out.detach().cpu().numpy())
        y_pred.extend(y_pred_batch.detach().cpu().numpy())
        y_true.extend(y_true_i.detach().cpu().numpy())
        losses.append(loss.item())
        
    y_pred_score = np.stack(y_pred_score)
    evaluate_metrics(args, y_true, y_pred_score, y_pred, mode=f'Training_split_{split_id}_{args.feature_removal_level}', n_classes=n_classes)


def validation(args, model, loss_fn, data, length_tensor, time_tensor, y_true, batch_size, n_classes, split_id):
    model.eval()
    with torch.no_grad():
        num_batch = (data.shape[1] // batch_size) + 1
        losses = []
        y_pred = []
        y_pred_score = []
        for b_i in trange(num_batch):
            out = model.forward(data[:, b_i*batch_size:(b_i+1)*batch_size, :], 
                                time = time_tensor[b_i*batch_size:(b_i+1)*batch_size], 
                                seq_len = length_tensor[b_i*batch_size:(b_i+1)*batch_size])
            
            loss = loss_fn(out, y_true[b_i*batch_size:(b_i+1)*batch_size])
            out = torch.nn.functional.softmax(out, dim=1)
            y_pred_score.extend(out.detach().cpu().numpy())
            _, y_pred_batch = torch.max(out, dim=1)
            y_pred.extend(y_pred_batch.detach().cpu().numpy())
            losses.append(loss.item())
        
        y_true = y_true.detach().cpu().numpy()
        losses = np.mean(losses)
        y_pred_score = np.stack(y_pred_score)
        val_metrics = evaluate_metrics(args, y_true, y_pred_score, y_pred, mode=f'Validation_split_{split_id}_{args.feature_removal_level}', n_classes=n_classes)
        return val_metrics



def one_testing(args, model_path, data, d_inp, d_hid, n_layers, n_head, length_tensor, time_tensor, y_true, batch_size, n_classes, max_len, split_id):
    model = Transformer_Modi(d_inp=d_inp, 
                             d_hid=d_hid, 
                             n_head=n_head,
                             max_len = max_len,
                             n_classes=n_classes,
                             n_layers=n_layers
                             ).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        num_batch = (data.shape[1] // batch_size) + 1
        y_pred = []
        y_pred_score = []
        for b_i in trange(num_batch):
            out = model.forward(data[:, b_i*batch_size:(b_i+1)*batch_size, :], 
                                time = time_tensor[b_i*batch_size:(b_i+1)*batch_size], 
                                seq_len=length_tensor[b_i*batch_size:(b_i+1)*batch_size])
            
            out = torch.nn.functional.softmax(out, dim=1)
            _, y_pred_batch = torch.max(out, dim=1)
            y_pred_score.extend(out.detach().cpu().numpy())
            y_pred.extend(y_pred_batch.detach().cpu().numpy())
            
        y_true = y_true.detach().cpu().numpy()
        y_pred_score = np.stack(y_pred_score)
        evaluate_metrics(args, y_true, y_pred_score, y_pred, mode=f'Testing_{args.feature_removal_level}', n_classes=n_classes)



def observe_testing(args, model, loss_fn, data, length_tensor, time_tensor, y_true, batch_size, n_classes, split_id):
    model.eval()
    with torch.no_grad():
        num_batch = (data.shape[1] // batch_size) + 1
        y_pred = []
        y_pred_score = []
        for b_i in trange(num_batch):
            out = model.forward(data[:, b_i*batch_size:(b_i+1)*batch_size, :], 
                                time = time_tensor[b_i*batch_size:(b_i+1)*batch_size], 
                                seq_len=length_tensor[b_i*batch_size:(b_i+1)*batch_size])
            
            out = torch.nn.functional.softmax(out, dim=1)
            _, y_pred_batch = torch.max(out, dim=1)
            y_pred_score.extend(out.detach().cpu().numpy())
            y_pred.extend(y_pred_batch.detach().cpu().numpy())
            
        y_true = y_true.detach().cpu().numpy()
        y_pred_score = np.stack(y_pred_score)
        evaluate_metrics(args, y_true, y_pred_score, y_pred, mode=f'Observe_Testing_split_{split_id}_{args.feature_removal_level}', n_classes=n_classes)


def evaluate_metrics(args, y_true, y_pred_score, y_pred, mode='Training', n_classes=2):
    base_path = './results'
    filename = f'{mode}_{args.dataset}_{args.missingratio}_results.csv'
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



def sample_data(args, Pval_tensor, Ptest_tensor):
    num_all_features =int(Pval_tensor.shape[2] / 2)
    num_missing_features = round(args.missingratio * num_all_features)
    if args.feature_removal_level == 'sample':
        for i, patient in enumerate(Pval_tensor):
            idx = np.random.choice(num_all_features, num_missing_features, replace=False)
            patient[:, idx] = torch.zeros(Pval_tensor.shape[1], num_missing_features)
            Pval_tensor[i] = patient
        for i, patient in enumerate(Ptest_tensor):
            idx = np.random.choice(num_all_features, num_missing_features, replace=False)
            patient[:, idx] = torch.zeros(Ptest_tensor.shape[1], num_missing_features)
            Ptest_tensor[i] = patient
    elif args.feature_removal_level == 'set':
        density_score_indices = np.load('./IG_density_scores_' + args.dataset + '.npy', allow_pickle=True)[:, 0]
        idx = density_score_indices[:num_missing_features].astype(int)
        Pval_tensor[:, :, idx] = torch.zeros(Pval_tensor.shape[0], Pval_tensor.shape[1], num_missing_features)
        Ptest_tensor[:, :, idx] = torch.zeros(Ptest_tensor.shape[0], Ptest_tensor.shape[1], num_missing_features)



####### Data Preprocessing Followed Raindrop #######
for k in range(n_splits):
    split_idx = k + 1
    print('Split id: %d' % split_idx)
    if args.dataset == 'P12':
        if subset == True:
            split_path = '/splits/phy12_split_subset' + str(split_idx) + '.npy'
        else:
            split_path = '/splits/phy12_split' + str(split_idx) + '.npy'
    elif args.dataset == 'P19':
        split_path = '/splits/phy19_split' + str(split_idx) + '_new.npy'
    elif args.dataset == 'eICU':
        split_path = '/splits/eICU_split' + str(split_idx) + '.npy'
    elif args.dataset == 'PAM':
        split_path = '/splits/PAM_split_' + str(split_idx) + '.npy'


    # prepare the data:
    Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path, split_type=args.splittype, reverse=reverse,
                                                                baseline=False, dataset=args.dataset,
                                                                predictive_label=args.predictive_label)
    
    if args.dataset == 'P12' or args.dataset == 'P19' or args.dataset == 'eICU':
        T, F = Ptrain[0]['arr'].shape
        D = len(Ptrain[0]['extended_static'])
        
        Ptrain_tensor = np.zeros((len(Ptrain), T, F))
        Ptrain_static_tensor = np.zeros((len(Ptrain), D))

        for i in range(len(Ptrain)):
            Ptrain_tensor[i] = Ptrain[i]['arr']
            Ptrain_static_tensor[i] = Ptrain[i]['extended_static']

        mf, stdf = getStats(Ptrain_tensor)
        ms, ss = getStats_static(Ptrain_static_tensor, dataset=args.dataset)
        
        Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor, Ptrain_length_tensor = tensorize_normalize(Ptrain, ytrain, mf,
                                                                                                        stdf, ms, ss)
        Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor, Pval_length_tensor = tensorize_normalize(Pval, yval, mf, stdf, ms, ss)
        Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor, Ptest_length_tensor = tensorize_normalize(Ptest, ytest, mf, stdf, ms,ss)
        if args.missingratio > 0:
            sample_data(args, Pval_tensor, Ptest_tensor)
            
    elif args.dataset == 'PAM':                                                                         
        T, F = Ptrain[0].shape
        D = 1
        
        Ptrain_tensor = Ptrain
        Ptrain_static_tensor = np.zeros((len(Ptrain), D))
        mf, stdf = getStats(Ptrain)
        Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor, Ptrain_length_tensor = tensorize_normalize_other(Ptrain, ytrain, mf, stdf)
        Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor, Pval_length_tensor = tensorize_normalize_other(Pval, yval, mf, stdf)
        Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor, Ptest_length_tensor = tensorize_normalize_other(Ptest, ytest, mf, stdf)
        if args.missingratio > 0:
            sample_data(args, Pval_tensor, Ptest_tensor)


    ytrain_tensor = ytrain_tensor.to(DEVICE)
    yval_tensor = yval_tensor.to(DEVICE)
    ytest_tensor = ytest_tensor.to(DEVICE)
    
    Ptrain_tensor = Ptrain_tensor.permute(1, 0, 2).to(DEVICE)
    Pval_tensor = Pval_tensor.permute(1, 0, 2).to(DEVICE)
    Ptest_tensor = Ptest_tensor.permute(1, 0, 2).to(DEVICE)
    
    fea_size = Ptrain_tensor.shape[2] // 2
    Ptrain_tensor = Ptrain_tensor[:, :, :fea_size].to(DEVICE)
    Pval_tensor = Pval_tensor[:, :, :fea_size].to(DEVICE)
    Ptest_tensor = Ptest_tensor[:, :, :fea_size].to(DEVICE)
    
    Ptrain_time = Ptrain_time_tensor.squeeze(2).to(DEVICE).to(torch.int32)
    Pval_time = Pval_time_tensor.squeeze(2).to(DEVICE).to(torch.int32)
    Ptest_time = Ptest_time_tensor.squeeze(2).to(DEVICE).to(torch.int32)
    
    Ptrain_length_tensor = torch.tensor(Ptrain_length_tensor).to(DEVICE).to(torch.int32)
    Pval_length_tensor = torch.tensor(Pval_length_tensor).to(DEVICE).to(torch.int32)
    Ptest_length_tensor = torch.tensor(Ptest_length_tensor).to(DEVICE).to(torch.int32)
    
    
    model = Transformer_Modi(d_inp=d_inp, 
                             d_hid=d_hid, 
                             max_len=max_len, 
                             n_classes=n_classes,
                             n_layers=n_layers,
                             n_head=n_head,
                             dropout=dropout).to(DEVICE)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True, factor=0.5, threshold=1e-4, patience=2)
    n_epoch = 50
    max_metric = 0
    for e in range(n_epoch):
        # Training  
        if n_classes == 2:
            Binary_training(args = args, 
                    model = model, 
                    loss_fn = loss_fn, 
                    optimizer = optimizer,
                    data = Ptrain_tensor, 
                    length_tensor = Ptrain_length_tensor,
                    time_tensor = Ptrain_time, 
                    y_true = ytrain_tensor, 
                    batch_size = batch_size, 
                    n_classes = n_classes,
                    split_id = split_idx)
        else:
            Multiclass_training(args = args, 
                                model = model, 
                                loss_fn = loss_fn, 
                                optimizer = optimizer,
                                data = Ptrain_tensor, 
                                length_tensor = Ptrain_length_tensor,
                                time_tensor = Ptrain_time, 
                                y_true = ytrain_tensor, 
                                batch_size = batch_size, 
                                n_classes = n_classes,
                                split_id = split_idx)
        # Validation
        val_metrics = validation(args = args,
                                 model = model, 
                                 loss_fn = loss_fn,
                                 data = Pval_tensor, 
                                 length_tensor = Pval_length_tensor, 
                                 time_tensor = Pval_time, 
                                 y_true = yval_tensor, 
                                 batch_size = batch_size, 
                                 n_classes = n_classes,
                                 split_id = split_idx)
        
        cur_metric = val_metrics[save_critiria].values[-1]
        if max_metric < cur_metric:
            directory = './best_model'
            model_savepath = os.path.join(directory, f'Multi-view-former_{args.dataset}_split_{split_idx}_{args.feature_removal_level}.pth')
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            torch.save(model.state_dict(), model_savepath)
            max_metric = cur_metric
        
        # Testing
        observe_testing(args = args,
                model = model, 
                loss_fn = loss_fn,
                data = Ptest_tensor, 
                length_tensor = Ptest_length_tensor, 
                time_tensor = Ptest_time, 
                y_true = ytest_tensor, 
                batch_size = batch_size, 
                n_classes = n_classes,
                split_id = split_idx)

    del model
    one_testing(args = args,
                model_path = model_savepath,
                d_inp = d_inp,
                d_hid = d_hid,
                n_head = n_head,
                n_layers = n_layers,
                data = Ptest_tensor, 
                length_tensor = Ptest_length_tensor, 
                time_tensor = Ptest_time, 
                y_true = ytest_tensor, 
                batch_size = batch_size, 
                n_classes = n_classes,
                max_len = max_len,
                split_id = split_idx)
    
# scheduler.step(val_metrics['f1'].values[0])
