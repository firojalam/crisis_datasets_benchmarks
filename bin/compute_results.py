#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 10:24:54 2018

@author: Firoj Alam

"""

import sys
import os
import optparse
import datetime
import numpy as np
from sklearn import metrics
import sklearn.metrics as metrics
from sklearn import preprocessing
import pandas as pd
import re

def classifaction_report(report):
    report_data = []
    lines = report.split('\n')
    #    print lines

    for line in lines[2:-3]:
        # print line
        line = line.strip()
        row = {}
        row_data = re.split('\s+', line)
        #        print row_data
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    (P, R, F1, sumClassCnt) = (0, 0, 0, 0)

    for row in report_data:
        tmp = row['precision']
        P = P + (tmp * row['support'])
        tmp = row['recall']
        R = R + (tmp * row['support'])
        tmp = row['f1_score']
        F1 = F1 + (tmp * row['support'])
        sumClassCnt = sumClassCnt + row['support']
    precision = P / sumClassCnt;
    recall = R / sumClassCnt;
    f1_score = F1 / sumClassCnt;
    print(str(precision) + "\t" + str(recall) + "\t" + str(f1_score) + "\n")
    return precision, recall, f1_score

if __name__ == '__main__':
    a = datetime.datetime.now().replace(microsecond=0)
    parser = optparse.OptionParser()
    parser.add_option('-i', action="store", dest="infile")
    options, args = parser.parse_args()

    file_name = options.infile
    actual=[]
    predicted=[]
    with open(file_name, 'rU') as f:
        for line in f:
            line = line.strip()
            if("inst#" in line):
                break
        for line in f:
            line = line.strip()
            # print line
            if (line==""):
                continue
            arr=line.split()
            actual_lab = arr[1].strip().split(":")[1]
            actual.append(actual_lab)
            predicted_lab = arr[2].strip().split(":")[1]
            predicted.append(predicted_lab)

    actual=np.array(actual)
    predicted = np.array(predicted)
    acc = precision = recall = f1_score = 0.0
    report = ""
    try:
        acc = metrics.accuracy_score(actual, predicted)*100
        # report = metrics.classification_report(actual, predicted)
        # precision, recall, f1_score = classifaction_report(report)
        precision = metrics.precision_score(actual, predicted, average="weighted")*100
        recall = metrics.recall_score(actual, predicted, average="weighted")*100
        f1_score = metrics.f1_score(actual, predicted, average="weighted")*100
    except Exception as e:
        print (e)
    pass
    result = str("{0:.2f}".format(acc)) + "\t" + str(
        "{0:.2f}".format(precision)) + "\t" + str("{0:.2f}".format(recall)) + "\t" + str(
        "{0:.2f}".format(f1_score))
    print(result)

    b = datetime.datetime.now().replace(microsecond=0)
    #print("Time taken: " + str((b - a)))