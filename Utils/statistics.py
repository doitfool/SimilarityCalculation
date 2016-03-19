# coding: utf-8

"""
    Project:    LSTM4CQA
    File:   statistic
    Author: AC
    Date:   2016/3/4 11:26
    Description: 计算准确率，精确度，召回率和F1
"""

import numpy as np


# 准确率
def accuracy_score(y_true, y_pred):
    return 1.0*(((y_true == 1)*(y_pred == 1)).sum() + ((y_true == 0)*(y_pred == 0)).sum()) / y_true.size


# 精确度
def precision_score(y_true, y_pred):
    return 1.0*((y_true == 1)*(y_pred == 1)).sum()/(y_pred == 1).sum()


# 召回率
def recall_score(y_true, y_pred):
    return 1.0*((y_true == 1)*(y_pred == 1)).sum()/(y_true == 1).sum()


# F1度量值
def f1_score(y_true, y_pred):
    return 2.0*precision_score(y_true, y_pred)*recall_score(y_true, y_pred) / ((precision_score(y_true, y_pred)+recall_score(y_true, y_pred)))

if __name__ == '__main__':
    y_test = np.array([0,1,0,1,1])

    classes = [np.array([0]), np.array([0]), np.array([0]), np.array([1]), np.array([1])]

    # b = []
    # for a in classes:
    #     b += list(a)
    # print b, np.array(b)

    classes = np.array(reduce(lambda x, y: x+y, [list(a) for a in classes]))

    # print np.array([list(a) for a in classes])
    # np

    print 'accuracy:', accuracy_score(y_test, classes)
    print 'precision:', precision_score(y_test, classes)
    print 'recall:', recall_score(y_test, classes)
    print 'F1:', f1_score(y_test, classes)

