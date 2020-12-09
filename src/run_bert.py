#!/usr/bin/env python
# coding: utf-8

import os
import gc
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

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
    def __init__(self, files, labels, tokenizer, max_len):
        self.files = files
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.len = len(labels)
        
    def __getitem__(self, index):
        files = self.files[index]
        labels = self.labels[index]
        return files, labels
    
    def __len__(self):
        return self.len
    
    def collate_fn(self, batch):
        files = []
        labels = []
        for item in batch:
            files.append(item[0])
            labels.append(item[1])
        # input_ids, token_type_ids, attention_mask
        inputs = self.tokenizer(files, padding=True, truncation=True, 
                                max_length=self.max_len, return_tensors='pt')
        labels = torch.LongTensor(labels)
        return inputs, labels

class MyModel(nn.Module):
    """自定义模型"""
    def __init__(self, bert, num_label):
        super(MyModel, self).__init__()
        self.bert = bert
        self.fc = nn.Sequential(
            nn.Linear(bert.config.hidden_size*3, bert.config.hidden_size),
            nn.Dropout(0.1),
            nn.Tanh(),
            nn.Linear(bert.config.hidden_size, num_label),
        )
        self.init_params()
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        last_hidden_state = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)['last_hidden_state']
        last_hidden_state = last_hidden_state.permute(0, 2, 1).contiguous()
        h0, last_hidden_state = last_hidden_state[:, :, 0], last_hidden_state[:, :, 1:]
        h1 = nn.MaxPool1d(last_hidden_state.shape[-1])(last_hidden_state).squeeze(-1)
        h2 = nn.AvgPool1d(last_hidden_state.shape[-1])(last_hidden_state).squeeze(-1)
        h = torch.cat([h0, h1, h2], dim=1)
        logits = self.fc(h)
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
        for inputs, labels in dataloader:
            for key in inputs:
                inputs[key] = inputs[key].cuda()
            logits = model(**inputs)
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

def model_train(args, df_train, df_test):
    """模型训练"""
    start_time = time.time()
    num_label = args.nlabels
    bert_path = args.bertpath
    max_length = args.maxlen
    n_splits = args.nfolds
    num_epochs = args.epochs
    num_warmup_steps = args.warmup
    batch_size = args.batchsize
    lr = args.lr
    
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    sub_dataset = MyDataset(df_test.tokens.values, df_test.label.values, tokenizer, max_length)
    sub_dataloader = DataLoader(sub_dataset, batch_size*10, shuffle=False, collate_fn=sub_dataset.collate_fn, num_workers=16, pin_memory=True)
    
    oof = np.zeros(shape=(df_train.shape[0], num_label))
    sub = np.zeros(shape=(df_test.shape[0], num_label))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)
    
    for i, (trn_idx, val_idx) in enumerate(skf.split(df_train, df_train.label)):
        print('--------------------------------- {} fold ---------------------------------'.format(i+1))
        trn_df = df_train.iloc[trn_idx]
        val_df = df_train.iloc[val_idx]
        trn_dataset = MyDataset(trn_df.tokens.values, trn_df.label.values, tokenizer, max_length)
        val_dataset = MyDataset(val_df.tokens.values, val_df.label.values, tokenizer, max_length)
        trn_dataloader = DataLoader(trn_dataset, batch_size, shuffle=True, collate_fn=trn_dataset.collate_fn, num_workers=16, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size*10, shuffle=False, collate_fn=val_dataset.collate_fn, num_workers=16, pin_memory=True)
        
        bert = BertModel.from_pretrained(bert_path, return_dict=True)
        model = MyModel(bert, num_label)
        if args.multigpu==1:
            model = nn.DataParallel(model)
        model = model.cuda()
        loss_func = CrossEntropyLabelSmooth(num_label)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_epochs*len(trn_dataloader))
        for epoch in range(num_epochs):
            # train
            model.train()
            trn_loss, ac_num, n = 0, 0, 0
            for inputs, labels in tqdm(trn_dataloader, desc='[epoch {:02d}/{:02d}]'.format(epoch+1, num_epochs)):
                for key in inputs:
                    inputs[key] = inputs[key].cuda()
                labels = labels.cuda()
                logits = model(**inputs)
                loss = loss_func(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                scheduler.step()
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
        # predict
        oof[val_idx] = model_predict(model, val_dataloader)
        sub += model_predict(model, sub_dataloader) / skf.n_splits
    print('finish training, cost {:.2f} min, cv_acc : {:.5f}'.format(
        (time.time()-start_time)/60, accuracy_score(df_train.label, oof.argmax(axis=1))))
    return oof, sub

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', default='./', type=str)
    parser.add_argument('--bertpath', default='./roberta_wwm/', type=str)
    parser.add_argument('--savepath', default='./output/', type=str)
    parser.add_argument('--multigpu', default=1, type=int)
    parser.add_argument('--nlabels', default=20, type=int)
    parser.add_argument('--maxlen', default=100, type=int)
    parser.add_argument('--nfolds', default=5, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--warmup', default=200, type=int)
    parser.add_argument('--batchsize', default=128, type=int)
    parser.add_argument('--lr', default=2.3e-5, type=float)
    args = parser.parse_args()
    set_all_seed(2020)
    start_time = time.time()
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
    
    print('loading data ...')
    # cols: [filepath, label, type, filename, content]
    df_train = pd.read_csv(args.datapath+'train.csv')
    df_test = pd.read_csv(args.datapath+'test2.csv')
    lbl = LabelEncoder().fit(df_train.label)
    df_train['label'] = lbl.transform(df_train.label)
    df_test['label']  = -1
    
    print('start training ...')
    # 只使用文件名进行训练, 对mask的样本设置默认概率分布。
    df_train['tokens'] = df_train.filename
    df_test['tokens'] = df_test.filename
    oof, sub = model_train(args, df_train, df_test)
    
    default = (df_train.label.value_counts().sort_index() / df_train.shape[0]).values
    oof1 = oof.copy()
    oof2 = oof.copy()
    
    oof1[df_train['mask']==1] = default
    oof2[df_train['mask']==0] = default
    sub[df_test['mask']]  = default
    
    np.save(args.savepath+'bert_oof0_1.npy', oof1)
    np.save(args.savepath+'bert_oof0_2.npy', oof2)
    np.save(args.savepath+'bert_sub0.npy', sub)
    
    # 只使用文件内容开头部分
    df_train['tokens'] = df_train.content.apply(lambda x: x[:args.maxlen])
    df_test['tokens'] = df_test.content.apply(lambda x: x[:args.maxlen])
    oof, sub = model_train(args, df_train, df_test)
    
    np.save(args.savepath+'bert_oof1.npy', oof)
    np.save(args.savepath+'bert_sub1.npy', sub)
    
    print('end, cost {:.5f} min'.format((time.time()-start_time)/60))