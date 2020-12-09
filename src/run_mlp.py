#!/usr/bin/env python
# coding: utf-8

import os
import gc
import time
import pkuseg
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pandarallel import pandarallel
pandarallel.initialize()
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def set_all_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

class MyDataset(Dataset):
    """自定义数据集"""
    def __init__(self, feats, labels):
        self.feats = feats
        self.labels = labels
        self.len = len(labels)
    
    def __getitem__(self, index):
        feat = self.feats[index]
        label = self.labels[index]
        return feat, label
    
    def __len__(self):
        return self.len

class MyModel(nn.Module):
    """自定义模型"""
    def __init__(self, input_size, num_label):
        super(MyModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, input_size//2),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size//2, input_size//4),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size//4, input_size//8),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size//8, num_label),
        )
        self.init_params()
    
    def forward(self, feats):
        logits = self.fc(feats)
        return logits
    
    def init_params(self):
        for name, w in self.fc.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(w)
            else:
                nn.init.zeros_(w)

class CrossEntropyLabelSmooth(nn.Module):
    """标签平滑Softmax"""
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + \
            self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

def model_predict(model, dataloader, val=False):
    """模型预测"""
    model.eval()
    prob = []
    if val:
        val_loss, ac_num, n = 0, 0, 0
        loss_func = nn.CrossEntropyLoss()
    with torch.no_grad():
        for feats, labels in dataloader:
            feats = feats.cuda()
            logits = model(feats)
            if val:
                labels = labels.cuda()
                loss = loss_func(logits, labels)
                val_loss += loss.item() * labels.shape[0]
                n += labels.shape[0]
                preds = F.softmax(logits, dim=1).argmax(dim=1)
                ac_num += (preds==labels).sum().item()
            prob.append(F.softmax(logits, dim=1).cpu().numpy())
    prob = np.concatenate(prob)
    if val:
        return prob, val_loss/n, ac_num/n
    else:
        return prob
    
class EarlyStopping:
    """早停"""
    def __init__(self, early_stop_round, model_path):
        self.early_stop_round = early_stop_round
        self.model_path = model_path
        
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        if val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.early_stop_round:
                self.early_stop = True
        else:
            self.counter = 0
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.model_path)

def get_feats(args, sentences):
    """构造特征"""
    X_tfidf = TfidfVectorizer(max_df=0.5, max_features=args.tfidfsize).fit_transform(sentences).astype('float32').toarray()
    X_count = CountVectorizer(max_df=0.5, max_features=args.tfidfsize).fit_transform(sentences).astype('float32').toarray()
    X = np.concatenate([X_tfidf, X_count], axis=1)
    return X

def model_train(args, df):
    """模型训练"""
    start_time = time.time()
    X = get_feats(args, df['tokens'])
    X_train = X[df['train']]
    X_test = X[~df['train']]
    
    input_size = X_train.shape[1]
    num_label = args.nlabels
    n_splits = args.nfolds
    num_epochs = args.epochs
    batch_size = args.batchsize
    lr = args.lr
    
    lbl = LabelEncoder().fit(df[df['train']].label)
    Y_train = lbl.transform(df[df['train']].label)
    Y_test = lbl.transform(df[~df['train']].label)
    
    sub_dataset = MyDataset(X_test, Y_test)
    sub_dataloader = DataLoader(sub_dataset, batch_size*10, shuffle=False, num_workers=16, pin_memory=True)
    
    oof = np.zeros(shape=(len(Y_train), num_label))
    sub = np.zeros(shape=(len(Y_test), num_label))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)
    
    for i, (trn_idx, val_idx) in enumerate(skf.split(X_train, Y_train)):
        print('--------------------------------- {} fold ---------------------------------'.format(i+1))
        trn_x, trn_y = X_train[trn_idx], Y_train[trn_idx]
        val_x, val_y = X_train[val_idx], Y_train[val_idx]
        trn_dataset = MyDataset(trn_x, trn_y)
        val_dataset = MyDataset(val_x, val_y)
        trn_dataloader = DataLoader(trn_dataset, batch_size, shuffle=True, num_workers=16, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size*10, shuffle=False, num_workers=16, pin_memory=True)
        
        model = MyModel(input_size, num_label)
        model = model.cuda()
        loss_func = CrossEntropyLabelSmooth(num_label)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, num_epochs, eta_min=2e-5)
        earlystop = EarlyStopping(early_stop_round=3, model_path='mlp.ckpt')
        for epoch in range(num_epochs):
            # train
            model.train()
            trn_loss, ac_num, n = 0, 0, 0
            for feats, labels in tqdm(trn_dataloader, desc='[epoch {:02d}/{:02d}]'.format(epoch+1, num_epochs)):
                feats = feats.cuda()
                labels = labels.cuda()
                logits = model(feats)
                loss = loss_func(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    trn_loss += loss.item()*labels.shape[0]
                    n += labels.shape[0]
                    preds = F.softmax(logits, dim=1).argmax(dim=1)
                    ac_num += (preds == labels).sum().item()
            trn_loss /= n
            trn_acc = ac_num/n
            # validation
            val_prob, val_loss, val_acc = model_predict(model, val_dataloader, val=True)
            print('[epoch {:02d}/{:02d}] | train_loss {:.5f} | train_acc {:.5f} | val_loss {:.5f} | val_acc {:.5f}'.format(epoch+1, num_epochs, trn_loss, trn_acc, val_loss, val_acc))
            scheduler.step()
            earlystop(val_loss, model)
            if earlystop.early_stop:
                break
        model.load_state_dict(torch.load(earlystop.model_path))
        # predict
        oof[val_idx] = model_predict(model, val_dataloader)
        sub += model_predict(model, sub_dataloader) / skf.n_splits
    
    print('finish training, cost {:.2f} min, cv_acc : {:.5f}'.format(
        (time.time()-start_time)/60, accuracy_score(Y_train, oof.argmax(axis=1))))
    return oof, sub

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', default='./', type=str)
    parser.add_argument('--savepath', default='./output/', type=str)
    parser.add_argument('--tfidfsize', default=15000, type=int)
    parser.add_argument('--nlabels', default=20, type=int)
    parser.add_argument('--nfolds', default=5, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batchsize', default=128, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    args = parser.parse_args()
    set_all_seed(2020)
    start_time = time.time()
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
    
    print('loading data ...')
    # cols: [filepath, label, type, filename, content]
    df_train = pd.read_csv(args.datapath+'train.csv')
    df_test = pd.read_csv(args.datapath+'test2.csv')
    df_train['train'] = True
    df_test['train'] = False
    df = pd.concat([df_train, df_test], ignore_index=True)
    gc.collect()
    
    seg = pkuseg.pkuseg()
    print('start training ...')
    # 只使用文件名进行训练, 对mask的样本设置默认概率分布
    df['tokens'] = df.filename.parallel_apply(lambda x: ' '.join(seg.cut(x)))
    oof, sub = model_train(args, df)
    
    default = (df_train.label.value_counts().sort_index() / df_train.shape[0]).values
    oof1 = oof.copy()
    oof2 = oof.copy()
    
    oof1[df_train['mask']==1] = default
    oof2[df_train['mask']==0] = default
    sub[df_test['mask']]  = default
    
    np.save(args.savepath+'mlp_oof0_1.npy', oof1)
    np.save(args.savepath+'mlp_oof0_2.npy', oof2)
    np.save(args.savepath+'mlp_sub0.npy', sub)
    
    # 只使用文件内容
    df['tokens'] = df.content.parallel_apply(lambda x: ' '.join(seg.cut(x)))
    oof, sub = model_train(args, df)
    
    np.save(args.savepath+'mlp_oof1.npy', oof)
    np.save(args.savepath+'mlp_sub1.npy', sub)
    
    print('end, cost {:.5f} min'.format((time.time()-start_time)/60))