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
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import torch
from torch import nn
from torch.nn import utils, init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

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
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels
        self.len = len(labels)
        
    def __getitem__(self, index):
        sentence = self.sentences[index]
        label = self.labels[index]
        return sentence, label
        
    def __len__(self):
        return self.len
    
    def collate_fn(self, batch):
        sentences = []
        labels = []
        for item in batch:
            sentences.append(torch.LongTensor(item[0]))
            labels.append(item[1])
        sentences = utils.rnn.pad_sequence(sentences, batch_first=True, padding_value=0)
        labels = torch.LongTensor(labels)
        inputs = {'sentences': sentences}
        return inputs, labels

class Inception(nn.Module):
    """Inception模块"""
    def __init__(self, cin, co, relu=True):
        super(Inception, self).__init__()
        assert(co%4 == 0)
        cos = co // 4
        self.branch1 = nn.Conv1d(cin, cos, 1)
        self.branch2 = nn.Sequential(
            nn.Conv1d(cin, cos, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(cos, cos, 3, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(cin, cos, 3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(cos, cos, 5,stride=1,padding=2),
        )
        self.branch4 = nn.Conv1d(cin,cos, 3, stride=1, padding=1)
    
    def forward(self,x):
        branch1=self.branch1(x)
        branch2=self.branch2(x)
        branch3=self.branch3(x)
        branch4=self.branch4(x)
        result=F.relu(torch.cat([branch1,branch2,branch3,branch4], dim=1))
        return result

class TextInceptionNet(nn.Module):
    """TextInception网络"""
    def __init__(self, embeddings, embedding_size, inception_size, num_label):
        super(TextInceptionNet, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.conv = nn.Sequential(
            Inception(embedding_size, inception_size),
            Inception(inception_size, inception_size),
        )
        self.fc = nn.Sequential(
            nn.Linear(inception_size, inception_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(inception_size, num_label),
        )
    
    def forward(self, sentences):
        # [batch, seq_ln]
        h = self.embed(sentences)
        # [batch, seq_ln, embedding_size]
        h = h.permute(0, 2, 1).contiguous()
        h = self.conv(h)
        # [batch, Inception_size, seq_ln]
        h = F.max_pool1d(h, h.shape[-1]).squeeze(-1)
        # [batch, Inception_size]
        logits = self.fc(h)
        # [batch, num_label]
        return logits

class TextCNN(nn.Module):
    """TextCNN"""
    def __init__(self, embeddings, embedding_size, hidden_size, num_label):
        super(TextCNN, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embeddings, freeze=True)
        kernel_sizes = [1,2,3,4]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embedding_size, hidden_size, k),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, hidden_size, k),
                nn.ReLU(inplace=True),
            ) for k in kernel_sizes
        ])
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * len(kernel_sizes), hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_label),
        )
    
    def forward(self, sentences):
        # [batch, seq_ln]
        h = self.embed(sentences)
        # [batch, seq_ln, embedding_size]
        h = h.permute(0, 2, 1).contiguous()
        # [batch, embedding_size, seq_len]
        h = [F.relu(conv(h)) for conv in self.convs]
        # n * [batch, hidden_size, seq_len]
        h = [F.max_pool1d(x, x.shape[-1]).squeeze(-1) for x in h]
        # n * [batch, hidden_size]
        h = torch.cat(h, dim=1)
        # [batch, hidden_size * n]
        logits = self.fc(h)
        return logits

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

def model_predict(model, dataloader, val=False):
    """模型预测"""
    model.eval()
    prob = []
    val_loss, ac_num, n = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            for key in inputs:
                inputs[key] = inputs[key].cuda()
            logits = model(**inputs)
            if val:
                labels = labels.cuda()
                loss = nn.CrossEntropyLoss()(logits, labels)
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

def model_train(args, df_train, df_test, embeddings):
    """模型训练"""
    start_time = time.time()
    embedding_size = args.embsize
    hidden_size = args.hiddensize
    num_label = args.nlabels
    n_splits = args.nfolds
    num_epochs = args.epochs
    batch_size = args.batchsize
    lr = args.lr
    
    sub_dataset = MyDataset(df_test.tokens.values, df_test.label.values)
    sub_dataloader = DataLoader(sub_dataset, batch_size*10, shuffle=False, collate_fn=sub_dataset.collate_fn, num_workers=16, pin_memory=True)
    
    oof = np.zeros(shape=(df_train.shape[0], num_label))
    sub = np.zeros(shape=(df_test.shape[0], num_label))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)
    
    for i, (trn_idx, val_idx) in enumerate(skf.split(df_train, df_train.label)):
        print('--------------------------------- {} fold ---------------------------------'.format(i+1))
        trn_df = df_train.iloc[trn_idx]
        val_df = df_train.iloc[val_idx]
        trn_dataset = MyDataset(trn_df.tokens.values, trn_df.label.values)
        val_dataset = MyDataset(val_df.tokens.values, val_df.label.values)
        trn_dataloader = DataLoader(trn_dataset, batch_size, shuffle=True, collate_fn=trn_dataset.collate_fn, num_workers=16, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size*10, shuffle=False, collate_fn=val_dataset.collate_fn, num_workers=16, pin_memory=True)
        
        model = TextCNN(embeddings, embedding_size, hidden_size, num_label)
        if args.multigpu==1:
            model = nn.DataParallel(model)
        model = model.cuda()
        loss_func = CrossEntropyLabelSmooth(num_label)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, num_epochs, eta_min=2e-5)
        earlystop = EarlyStopping(early_stop_round=3, model_path='cnn.ckpt')
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
        (time.time()-start_time)/60, accuracy_score(df_train.label, oof.argmax(axis=1))))
    return oof, sub

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', default='./', type=str)
    parser.add_argument('--savepath', default='./output/', type=str)
    parser.add_argument('--multigpu', default=1, type=int)
    parser.add_argument('--hiddensize', default=1024, type=int)
    parser.add_argument('--nlabels', default=20, type=int)
    parser.add_argument('--nfolds', default=5, type=int)
    parser.add_argument('--maxlen', default=400, type=int)
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
    lbl = LabelEncoder().fit(df_train.label)
    df_train['label'] = lbl.transform(df_train.label)
    df_test['label']  = -1
    
    vocab = np.load('./vocab.npy', allow_pickle=True).item()
    embeddings = torch.from_numpy(np.load('./w2v.npy'))
    args.embsize = embeddings.shape[1]
    
    seg = pkuseg.pkuseg()
    def word2idx(sentence, vocab, seg):
        idx = []
        for w in sentence:
            if w in vocab:
                idx.append(vocab[w])
        if len(idx) == 0:
            idx = [0]
        return np.array(idx)
    df_train['filename'] = df_train.filename.parallel_apply(lambda x: word2idx(x, vocab, seg)[:args.maxlen])
    df_train['content'] = df_train.content.parallel_apply(lambda x: word2idx(x, vocab, seg)[:args.maxlen])
    df_test['filename'] = df_test.filename.parallel_apply(lambda x: word2idx(x, vocab, seg)[:args.maxlen])
    df_test['content'] = df_test.content.parallel_apply(lambda x: word2idx(x, vocab, seg)[:args.maxlen])
    
    print('start training ...')
    
    # 只使用文件名进行训练, 对mask的样本设置默认概率分布
    df_train['tokens'] = df_train.filename
    df_test['tokens'] = df_test.filename
    oof, sub = model_train(args, df_train, df_test, embeddings)
    
    default = (df_train.label.value_counts().sort_index() / df_train.shape[0]).values
    oof1 = oof.copy()
    oof2 = oof.copy()
    
    oof1[df_train['mask']==1] = default
    oof2[df_train['mask']==0] = default
    sub[df_test['mask']]  = default
    
    np.save(args.savepath+'cnn_oof0_1.npy', oof1)
    np.save(args.savepath+'cnn_oof0_2.npy', oof2)
    np.save(args.savepath+'cnn_sub0.npy', sub)
    
    # 只使用文件内容
    df_train['tokens'] = df_train.content
    df_test['tokens'] = df_test.content
    oof, sub = model_train(args, df_train, df_test, embeddings)
    
    np.save(args.savepath+'cnn_oof1.npy', oof)
    np.save(args.savepath+'cnn_sub1.npy', sub)
    
    print('end, cost {:.5f} min'.format((time.time()-start_time)/60))