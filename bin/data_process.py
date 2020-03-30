# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 20:42:24 2017

@author: Firoj Alam
"""


import numpy as np
np.random.seed(1337)  # for reproducibility

import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import sys
from sklearn import preprocessing
import pandas as pd
import re
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
import math
stop_words = set(stopwords.words('english'))
import random
random.seed(1337)
import aidrtokenize as aidrtokenize

def file_exist(file_name):
    if os.path.exists(file_name):
        return True
    else:
        return False

def read_stop_words(file_name):
    if(not file_exist(file_name)):
        print("Please check the file for stop words, it is not in provided location "+file_name)
        sys.exit(0)
    stop_words =[]
    with open(file_name, 'rU') as f:
        for line in f:
            line = line.strip()
            if (line == ""):
                continue
            stop_words.append(line)
    return stop_words;

stop_words_file="etc/stop_words_english.txt"
stop_words = read_stop_words(stop_words_file)

 

def read_train_data(dataFile, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, delim):
    """
    Prepare the data
    """
    data=[]
    lab=[]
    with open(dataFile, 'rU') as f:
        next(f)    
        for line in f:
            line = line.strip()   
            if (line==""):
                continue                            		
            row=line.split(delim)
            txt = row[3].strip()
            txt = txt.replace("'", "")
            txt = aidrtokenize.tokenize(txt)

            label = row[6]
            txt = txt.replace("'", "")
            w_list=[]
            for w in txt.split():
                if w not in stop_words:
                    try:
                        #w=str(w.encode('ascii'))
                        w_list.append(w.encode('utf-8'))
                    except Exception as e:
                        print(w)
                        pass
            text = " ".join(w_list)

            # if(len(text)<1):
            #     print txt
            #     continue
            #txt=aidrtokenize.tokenize(txt)
            #txt=[w for w in txt if w not in stop_words]              
            if(isinstance(text, str)):
                data.append(text)
                lab.append(label)
            else:
                print(text)

    data_shuf = []
    lab_shuf = []
    index_shuf = range(len(data))
    random.shuffle(index_shuf)
    for i in index_shuf:
        data_shuf.append(data[i])
        lab_shuf.append(lab[i])


    le = preprocessing.LabelEncoder()
    yL=le.fit_transform(lab_shuf)
    labels=list(le.classes_)
    
    label=yL.tolist()
    yC=len(set(label))
    yR=len(label)
    y = np.zeros((yR, yC))
    y[np.arange(yR), yL] = 1
    y=np.array(y,dtype=np.int32)
    

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token="OOV_TOK")
    tokenizer.fit_on_texts(data_shuf)
    sequences = tokenizer.texts_to_sequences(data_shuf)
    
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    #labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    #print('Shape of label tensor:', labels.shape)    
    #return data,labels,word_index,dim;        
    return data,y,le,labels,word_index,tokenizer

    
def read_dev_data(dataFile, tokenizer, MAX_SEQUENCE_LENGTH, delim, train_le):
    """
    Prepare the data
    """      
    data=[]
    lab=[]
    with open(dataFile, 'rU') as f:
        next(f)    
        for line in f:
            line = line.strip()   
            if (line==""):
                continue                            		
            row=line.split(delim)
            txt = row[3].strip()
            txt = txt.replace("'", "")
            txt = aidrtokenize.tokenize(txt)

            label = row[6]

            txt = txt.replace("'","")
            w_list=[]
            for w in txt.split():
                if w not in stop_words:
                    try:
                        #w=str(w.encode('ascii'))
                        w_list.append(w.encode('utf-8'))
                    except Exception as e:
                        #print(w)
                        #print(e)
                        pass
            text = " ".join(w_list)

            # if(len(text)<1):
            #     print txt
            #     continue
            #txt=aidrtokenize.tokenize(txt)
            #txt=[w for w in txt if w not in stop_words]
            if(isinstance(text, str)):
                data.append(text)
                lab.append(label)
            else:
                print("not text: "+text)

    le = train_le #preprocessing.LabelEncoder()
    yL=le.transform(lab)
    labels=list(le.classes_)
    
    label=yL.tolist()
    yC=len(set(label))
    yR=len(label)
    y = np.zeros((yR, yC))
    y[np.arange(yR), yL] = 1
    y=np.array(y,dtype=np.int32)

    sequences = tokenizer.texts_to_sequences(data)   
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))    
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data.shape)
    return data,y,le,labels,word_index


def read_data_classifier(dataFile, tokenizer, MAX_SEQUENCE_LENGTH, delim, train_le):
    """
    Prepare the data
    """
    data = []
    lab = []
    with open(dataFile, 'rU') as f:
        next(f)
        for line in f:
            line = line.strip()
            if (line == ""):
                continue
            row = line.split(delim)
            txt = row[3].strip()
            txt = txt.replace("'", "")
            txt = aidrtokenize.tokenize(txt)

            label = row[6]

            txt = txt.replace("'", "")
            w_list = []
            for w in txt.split():
                if w not in stop_words:
                    try:
                        # w=str(w.encode('ascii'))
                        w_list.append(w.encode('utf-8'))
                    except Exception as e:
                        # print(w)
                        # print(e)
                        pass
            text = " ".join(w_list)

            # if(len(text)<1):
            #     print txt
            #     continue
            # txt=aidrtokenize.tokenize(txt)
            # txt=[w for w in txt if w not in stop_words]
            if (isinstance(text, str)):
                data.append(text)
                lab.append(label)
            else:
                print("not text: " + text)

    # le = train_le  # preprocessing.LabelEncoder()
    # yL = le.transform(lab)
    # labels = list(le.classes_)
    #
    # label = yL.tolist()
    # yC = len(set(label))
    # yR = len(label)
    # y = np.zeros((yR, yC))
    # y[np.arange(yR), yL] = 1
    # y = np.array(y, dtype=np.int32)

    sequences = tokenizer.texts_to_sequences(data)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data_x = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data_x.shape)
    return data_x, data, lab, #le, labels, word_index

def load_embedding(fileName):
    print('Indexing word vectors.')    
    embeddings_index = {}    
    f = open(fileName)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()    
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index;

def prepare_embedding(word_index, model, MAX_NB_WORDS, EMBEDDING_DIM):
    
    # prepare embedding matrix
    nb_words = min(MAX_NB_WORDS, len(word_index)+1)    
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM),dtype=np.float32)
    print(len(embedding_matrix))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        try:
            embedding_vector = model[word][0:EMBEDDING_DIM] #embeddings_index.get(word)
            embedding_matrix[i] = np.asarray(embedding_vector,dtype=np.float32)
        except KeyError:
            try:
                print(word +" not found... assigning zeros")
                rng = np.random.RandomState()        	
                #embedding_vector = rng.randn(EMBEDDING_DIM) #np.random.random(num_features)
                embedding_vector = np.zeros(EMBEDDING_DIM)  # np.random.random(num_features)
                embedding_matrix[i] = np.asarray(embedding_vector,dtype=np.float32)
            except KeyError:    
                continue      
    return embedding_matrix;

def str_to_indexes(s):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    input_size = 1014
    length = input_size
    alphabet_size = len(alphabet)
    char_dict = {}  # Maps each character to an integer
    # self.no_of_classes = num_of_classes
    for idx, char in enumerate(alphabet):
        char_dict[char] = idx + 1
    length = input_size


    """
    Convert a string to character indexes based on character dictionary.

    Args:
        s (str): String to be converted to indexes

    Returns:
        str2idx (np.ndarray): Indexes of characters in s

    """
    s = s.lower()
    max_length = min(len(s), length)
    str2idx = np.zeros(length, dtype='int64')
    for i in range(1, max_length + 1):
        c = s[-i]
        if c in char_dict:
            str2idx[i - 1] = char_dict[c]
    return str2idx
