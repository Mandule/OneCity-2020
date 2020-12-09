import gc
import time
import random
import argparse
import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    start_time = time.time()
    
    df_train = pd.read_csv('./train.csv')
    df_test = pd.read_csv('./test2.csv')
    lbl = LabelEncoder().fit(df_train.label)
    Y_train = lbl.transform(df_train.label)
    
    oof_files = ['bert_oof0_1', 'cnn_oof0_1', 'mlp_oof0_1', 
                 'bert_oof1', 'cnn_oof1', 'mlp_oof1']
    sub_files = ['bert_sub0', 'cnn_sub0', 'mlp_sub0',
                 'bert_sub1', 'cnn_sub1', 'mlp_sub1']
    oofs = []
    subs = []
    for f in oof_files:
        oofs.append(np.load('./output/'+f+'.npy'))
    for f in sub_files:
        subs.append(np.load('./output/'+f+'.npy'))
    
    X_train = np.concatenate(oofs, axis=1)
    X_test = np.concatenate(subs, axis=1)
    print(X_train.shape)
    print(X_test.shape)
    
    oof = np.zeros((X_train.shape[0],20))
    sub = np.zeros((X_test.shape[0],20))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
    
    for i, (trn_idx, val_idx) in enumerate(skf.split(X_train, Y_train)):
        print('-----------------{} fold-----------------'.format(i+1))
        X_trn, Y_trn = X_train[trn_idx], Y_train[trn_idx]
        X_val, Y_val = X_train[val_idx], Y_train[val_idx]
        X_sub = X_test
        clf = LGBMClassifier(
            objective='multiclass',
            num_leaves=63,
            learning_rate=0.01,
            n_estimators=10000,
            subsample_freq=1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            random_state=2020,
            n_jobs=24,
        )
        clf.fit(
            X_trn, Y_trn,
            eval_set=[(X_val, Y_val)],
            early_stopping_rounds=500,
            verbose=200,
        )
        print('val_acc: {:.5f}'.format(accuracy_score(Y_val, clf.predict(X_val))))
        oof[val_idx] = clf.predict_proba(X_val)
        sub += clf.predict_proba(X_sub) / skf.n_splits
        
    print('cv_acc : {:.5f}'.format(accuracy_score(Y_train, oof.argmax(axis=1))))
    print(classification_report(Y_train, oof.argmax(axis=1), target_names=lbl.classes_))
    
    oof_files = ['bert_oof0_2', 'cnn_oof0_2', 'mlp_oof0_2', 
                 'bert_oof1', 'cnn_oof1', 'mlp_oof1']
    sub_files = ['bert_sub0', 'cnn_sub0', 'mlp_sub0',
                 'bert_sub1', 'cnn_sub1', 'mlp_sub1']
    oofs = []
    subs = []
    for f in oof_files:
        oofs.append(np.load('./output/'+f+'.npy'))
    for f in sub_files:
        subs.append(np.load('./output/'+f+'.npy'))
    
    X_train = np.concatenate(oofs, axis=1)
    X_test = np.concatenate(subs, axis=1)
    print(X_train.shape)
    print(X_test.shape)
    
    oof = np.zeros((X_train.shape[0],20))
    
    for i, (trn_idx, val_idx) in enumerate(skf.split(X_train, Y_train)):
        print('-----------------{} fold-----------------'.format(i+1))
        X_trn, Y_trn = X_train[trn_idx], Y_train[trn_idx]
        X_val, Y_val = X_train[val_idx], Y_train[val_idx]
        X_sub = X_test
        clf = LGBMClassifier(
            objective='multiclass',
            num_leaves=63,
            learning_rate=0.01,
            n_estimators=10000,
            subsample_freq=1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            random_state=2020,
            n_jobs=24,
        )
        clf.fit(
            X_trn, Y_trn,
            eval_set=[(X_val, Y_val)],
            early_stopping_rounds=500,
            verbose=200,
        )
        print('val_acc: {:.5f}'.format(accuracy_score(Y_val, clf.predict(X_val))))
        oof[val_idx] = clf.predict_proba(X_val)
        sub += clf.predict_proba(X_sub) / skf.n_splits
    
    sub /= 2
    pd.DataFrame({
        'filename': df_test.filepath,
        'label': lbl.inverse_transform(sub.argmax(axis=1)),
    }).to_csv('./submission.csv', index=False)
    
    print('cv_acc : {:.5f}'.format(accuracy_score(Y_train, oof.argmax(axis=1))))
    print(classification_report(Y_train, oof.argmax(axis=1), target_names=lbl.classes_))
    print('end, cost {:.5f} min'.format((time.time()-start_time)/60))