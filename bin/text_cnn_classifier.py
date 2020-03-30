#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 10:24:54 2017

@author: firojalam
"""

import sys
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import load_model
import optparse
import datetime
from nltk.corpus import stopwords
import math
import performance as performance
import data_process as data_process
import os
from time import time
from datetime import datetime

stop_words = set(stopwords.words('english'))


class Instance(object):
    def __init__(self, id=1, date="", txtOrg="", txt=""):
        self.id = id
        self.date = date
        self.txtOrg = txtOrg
        self.txt = txt


def get_data(dataFile, tokenizer, MAX_SEQUENCE_LENGTH, delim):
    """
    Prepare the data
    """
    data = []
    label_list = []
    instances = []
    with open(dataFile, 'rU') as f:
        next(f)
        for line in f:
            line = line.strip()
            if (line == ""):
                continue
            row = line.split(delim)
            # ID = row[0].strip()
            # date = row[1].strip()
            # txtOrg = row[2].strip()
            txt = row[0].strip()
            label = row[1].strip()
            w_list = []
            for w in txt.split():
                if w not in stop_words:
                    try:
                        w = str(w.encode('ascii'))
                        w_list.append(w)
                    except Exception as e:
                        # print(w)
                        # print(e)
                        pass
            text = " ".join(w_list)
            if (len(text) < 1):
                print text
                continue
            data.append(text)
            instances.append(text)
            label_list.append(label)

    sequences = tokenizer.texts_to_sequences(data)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data.shape)
    return data, instances, label_list


def load_nn_model(model_file):
    loaded_model = load_model(model_file)
    print("Loaded model from disk")
    return loaded_model


def read_config(configfile):
    configdict = {}
    with open(configfile, 'rU') as f:
        for line in f:
            line = line.strip()
            if (line == ""):
                continue
            row = line.split("=")
            configdict[row[0]] = row[1]
    return configdict


def write2File(outfilename, prediction, probabilities, instances, label_list):
    text_file = open(outfilename, "w")
    text_file.write("Text\tPrediction\tGold\n");
    for lab_pred, prob, inst, lab_ref in zip(prediction, probabilities, instances, label_list):
        # if (lab_pred != lab_ref):
        tmpData = inst + "\t" + lab_pred + "\t" + lab_ref
        text_file.write(tmpData + "\n");
    text_file.close


if __name__ == '__main__':

    parser = optparse.OptionParser()
    parser.add_option('-c', action="store", dest="config_file")
    parser.add_option('-d', action="store", dest="data_file")
    parser.add_option('-l', action="store", dest="classified_file")
    parser.add_option('-o', action="store", dest="output_file")
    options, args = parser.parse_args()

    config_file = options.config_file
    data_file = options.data_file
    classified_file = options.classified_file


    delim = "\t"
    MAX_SEQUENCE_LENGTH = 25
    batch_size = 128

    configdict = read_config(config_file)

    loaded_model = load_nn_model(configdict["model_file"])
    tokenizer_file = configdict["tokenizer_file"]
    label_encoder_file = configdict["label_encoder_file"]

    # loading tokenizer
    with open(tokenizer_file, 'rb') as handle:
        tokenizer = pickle.load(handle)

    # loading label_encoder
    with open(label_encoder_file, 'rb') as handle:
        label_encoder = pickle.load(handle)

    # data, instances, label_list = get_data(data_file, tokenizer, MAX_SEQUENCE_LENGTH, delim)

    test_x, instances, label_list = data_process.read_data_classifier(data_file, tokenizer, MAX_SEQUENCE_LENGTH,delim,label_encoder)


    # classify data
    a = datetime.now().replace(microsecond=0)
    prediction = loaded_model.predict([test_x], batch_size=batch_size, verbose=1)
    b = datetime.now().replace(microsecond=0)
    print("Time taken for prediction: " + str((b - a)))
    print ("Data size: " + str(len(test_x)))

    probability_index = np.argmax(prediction, axis=1)
    probabilities = []
    for index, prob in zip(probability_index, prediction):
        probabilities.append(prob[index])

    class_labels = label_encoder.inverse_transform(probability_index)
    write2File(classified_file, class_labels, probabilities, instances, label_list)

    results_file = options.output_file
    out_file = open(results_file, "w")

    AUC, accu, P, R, F1, report = performance.performance_measure_classifier(label_list, prediction, label_encoder)

    # dir_name = os.path.dirname(classified_file)
    # base_name = os.path.basename(classified_file)
    base_name = os.path.splitext(data_file)[0]
    #dev_out_label_file_name = dir_name + "/" + base_name + "_dev_labels.txt"

    # accu = accu * 100
    # wauc=wAUC*100
    # auc=AUC*100
    precision = P * 100
    recall = R * 100
    f1_score = F1 * 100
    result = str("{0:.4f}".format(accu)) + "\t" + str("{0:.4f}".format(P)) + "\t" + str(
        "{0:.2f}".format(R)) + "\t" + str("{0:.4f}".format(F1)) + "\t" + str("{0:.4f}".format(AUC))+ "\n"
    print("results-cnn:\t"+base_name+"\t"+result)
    print (report)
    out_file.write(data_file+ "\n")
    out_file.write(result)
    out_file.write(report)

    conf_mat_str = performance.format_conf_mat_classifier(label_list, prediction, label_encoder)
    out_file.write(conf_mat_str+"\n")
    out_file.close()
    b = datetime.now().replace(microsecond=0)
    print ("time taken:")
    print(b - a)

