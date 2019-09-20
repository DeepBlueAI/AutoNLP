# distutils: language = c++
"""
Created on Mon Aug 26 11:04:11 2019
@author: Milk
@Concat: milk@pku.edu.cn
"""

import numpy as np
import time 
from cython cimport boundscheck,wraparound
import re
import jieba



'''
************************EN SEG1************************
'''
@boundscheck(False) 
@wraparound(False)
def clean_text_en_seg1(dat,sentence_len):
    cdef:
        int n = len(dat)
        int i = 0
    REPLACE_BY_SPACE_RE = re.compile('["/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')
    if sentence_len is None:
        ret = []
        for i in range(n):
            line = dat[i]
            line = line.lower()
            line = REPLACE_BY_SPACE_RE.sub(' ', line)
            line = BAD_SYMBOLS_RE.sub('', line)
            line = line.strip()
            line = line.split(' ')
            ret.append(line)
        return ret
        
    else:
        ret = []
        for i in range(n):
            line = dat[i]
            line = line[0:sentence_len]
            line = line.lower()
            line = REPLACE_BY_SPACE_RE.sub(' ', line)
            line = BAD_SYMBOLS_RE.sub('', line)
            line = line.strip()
            line = line.split(' ')
            ret.append(line)
        return ret




'''
************************EN SEG2************************
'''
@boundscheck(False) 
@wraparound(False)
def clean_text_en_seg2(str[:] dat,sentence_len):
    cdef:
        int n = len(dat)
        int i = 0
    REPLACE_BY_SPACE_RE = re.compile('["/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')
    
    ret = []
    for i in range(n):
        line = dat[i]
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        line = BAD_SYMBOLS_RE.sub('', line)
        line = line.strip()
        line = line.split(' ')
        ret.append(line)
    return ret
'''
************************ZH SEG1************************
'''

@boundscheck(False) 
@wraparound(False)
def clean_text_zh_seg1(str[:] dat,sentence_len):
    cdef:
        int n = len(dat)
        int i = 0
    REPLACE_BY_SPACE_RE = re.compile('[“”【】/（）：！～「」、|，；。"/(){}\[\]\|@,\.;]')
    if sentence_len is None:
        ret = []
        for i in range(n):
            line = dat[i]
            line = REPLACE_BY_SPACE_RE.sub('', line)
            ret.append(line)
        return ret
        
    else:
        ret = []
        for i in range(n):
            line = dat[i]
            line = line[0:sentence_len]
            line = REPLACE_BY_SPACE_RE.sub('', line)
            ret.append(line)
        return ret





'''
************************ZH SEG2************************
'''


def _tokenize_chinese_words(text):
    gen = jieba.cut(text, cut_all=False, HMM = False)
    ans = []
    for i in gen:
        ans.append(i)
    return ans
@boundscheck(False) 
@wraparound(False)
def clean_text_zh_seg2(dat,int sentence_len):
    cdef:
        int n = len(dat)
        int i = 0
    REPLACE_BY_SPACE_RE = re.compile('[“”【】/（）：！～「」、|，；。"/(){}\[\]\|@,\.;]')
    ret = []
    for i in range(n):
        line = dat[i]
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        line = line.strip()
        ret.append(_tokenize_chinese_words(line))
    return ret


'''
************************Sequentical************************
'''



@boundscheck(False) 
@wraparound(False)
def bulid_index(str[:] data,int num_sentence):
#    s1 = time.time()
    
    cdef:
        int min_df = 3
        int [:] text_lens = np.zeros( num_sentence,dtype=np.int32 )
        dict word2cnt = {}
        dict word2index = {}
        int i = 0
        int ind = 1
        
    for i in range(num_sentence):
        line = data[i]
        
        for w in line:
            if w in word2cnt:
                word2cnt[w] += 1
            else:
                word2cnt[w] = 1
        
        text_lens[i] = len(line)
    
    for k,v in word2cnt.items():
        if v >= min_df:
            word2index[k] = ind
            ind += 1
    MAX_VOCAB_SIZE = ind

    MAX_SEQ_LENGTH = np.sort(text_lens)[int(num_sentence*0.95)]


    return MAX_VOCAB_SIZE, MAX_SEQ_LENGTH, word2index,text_lens

@boundscheck(False) 
@wraparound(False)
def texts_to_sequences_and_pad(str[:] data,int num_sentence,dict word2index,int max_length,int[:] text_lens,int data_type):
    ans = np.zeros((num_sentence, max_length), dtype=np.int32)
    cdef:
        int k = len(word2index) + 1
        int [:,:] x_train = ans
        int i,j,n
    
    if data_type == 0:
        for i in range(num_sentence):
            line = data[i]
            n = min(max_length, text_lens[i])
            for j in range(n):
                w = line[j]
                if w in word2index:
                    x_train[i][j] = word2index[w]
                else:
                    x_train[i][j] = k
    else:
        for i in range(num_sentence):
            line = data[i]
            n = min(max_length, len(line))
            for j in range(n):
                w = line[j]
                if w in word2index:
                    x_train[i][j] = word2index[w]
                else:
                    x_train[i][j] = k
        
    return ans


	