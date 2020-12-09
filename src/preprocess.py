import re
import gc
import os
import time
import pkuseg
import random
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from pandarallel import pandarallel
import warnings
warnings.filterwarnings('ignore')
pandarallel.initialize()

def get_file_type(path):
    """获取文件的编码类型"""
    try:
        f = open(path, 'r', encoding='utf-8')
        line = f.read()
        f.close()
    except UnicodeDecodeError:
        return 'Excel'
    except OSError:
        return 'OSError'
    except:
        return 'OtherError'
    if '<div>' in line and '</div>' in line:
        return 'HTML'
    return 'CSV'

def get_file_name(s):
    """正则匹配文件名"""
    tokens = re.findall('[\u4e00-\u9fa5]+',s)
    if len(tokens) == 0:
        return '空'
    else:
        return ' '.join(tokens)

def get_file_content(path, file_type):
    """正则匹配文件内容"""
    def uniq(x):
        """去重复"""
        s = set()
        for i in x:
            if i in s:
                continue
            else:
                s.add(i)
                yield i
    
    if file_type in ['CSV', 'HTML']:
        # 文件类型为 utf-8
        with open(path, 'r') as f:
            content = ' '.join(f.readlines())
            tmp = ' '.join(uniq(re.findall('[\u4e00-\u9fa5]+', content)))
            if len(tmp) == 0:
                tmp = ' '.join(uniq(re.findall('[a-zA-Z]+', content)))
            content = tmp
    
    elif file_type == 'Excel':
        # 文件类型为 xls
        content = []
        for i in range(5):
            try:
                content.append(str(pd.read_excel(path, sheet_name=i)))
            except:
                pass
        content = ' '.join(content)
        tmp = ' '.join(uniq(re.findall('[\u4e00-\u9fa5]+', content)))
        if len(tmp) == 0:
            tmp = ' '.join(uniq(re.findall('[a-zA-Z]+', content)))
        content = tmp
    else:
        # 文件不可读
        content = '空'
    
    if len(content) == 0:
            content = '空'
    
    return content[:2000]

def mask_filename(content):
    """对训练集50%的内容非空文件的文件名进行mask"""
    if len(content) < 10:
        return -1
    else:
        if random.random() > 0.5:
            return 1
        else:
            return 0

if __name__ == '__main__':
    print('start preprocess ...')
    random.seed(2020)
    start_time = time.time()
    data_path = './data/'
    df_train = pd.read_csv(data_path+'answer_train.csv')
    df_test1 = pd.read_csv(data_path+'submit_example_test1.csv')
    df_test2 = pd.read_csv(data_path+'submit_example_test2.csv')
    shape = [df_train.shape[0], df_test1.shape[0]]
    df = pd.concat([df_train, df_test1, df_test2], ignore_index=True)
    
    df['filepath'] = df.filename
    df['type'] = df.filepath.parallel_apply(lambda s: get_file_type(data_path+s))
    df['content'] = df.parallel_apply(lambda s: get_file_content(data_path+s.filepath, s.type), axis=1)
    df['filename'] =  df.filepath.parallel_apply(get_file_name)
    
    # word2vec
    sentences = []
    seg = pkuseg.pkuseg()
    sentences += df.filename.parallel_apply(lambda x: seg.cut(x)).values.tolist()
    sentences += df.content.parallel_apply(lambda x: seg.cut(x)).values.tolist()
    emb_size = 256
    model = Word2Vec(sentences, size=emb_size, window=10, min_count=5, workers=32, sg=1)
    vocab = {'<pad>' : 0}
    for k, v in model.wv.vocab.items():
        vocab[k] = v.index + 1
    pad = np.array([[0.0]*emb_size]).astype(np.float32)
    vec = np.concatenate([pad, model.wv.vectors]).astype(np.float32)
    np.save('./w2v.npy', vec)
    np.save('./vocab.npy', vocab)
    
    df_train = df.iloc[:shape[0]].reset_index(drop=True)
    df_train['mask'] = df_train.content.parallel_apply(mask_filename)
    df_train.to_csv('./train.csv', index=False)
    
    df_test = df.iloc[shape[0]+shape[1]:].reset_index(drop=True)
    df_test['mask'] = df_test.filename.parallel_apply(lambda x: x == '空')
    df_test.to_csv('./test2.csv', index=False)
    
    print('finish preprocess, cost {:.2f} min'.format((time.time()-start_time)/60))