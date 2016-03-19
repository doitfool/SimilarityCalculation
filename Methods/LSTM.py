# coding: utf-8

"""
    Project:    LSTM4CQA
    File:   LSTM
    Author: AC
    Date:   2016/2/25 15:30
    Description:    
"""
import numpy as np
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleDeepRNN, SimpleRNN
from numpy.random import permutation
from Utils.FileProcess import generate_w2i_i2w_dict, sentence2sequence
from Utils.statistics import *

# 获取X, y
def get_data(data_file):
    X = []
    y = []
    w2i, i2w = generate_w2i_i2w_dict(data_file)
    with open(data_file, 'r') as fr:
        for line in fr:
            query1, query2, label = line.split('\t')[:3]
            query = (sentence2sequence(query1, w2i)+sentence2sequence(query2, w2i))
            X.append(query)
            y.append(int(label))
    X = np.array(X)
    y = np.array(y)
    return X, y

# 自定义阈值分类
def get_class(probas, threshold):
    classes = []
    for proba in probas:
        if proba[0] > threshold:
            classes.append(1)
        else:
            classes.append(0)
    classes = np.array(classes)
    return classes

def lstm(X, y, word_vec_len=300, batch_size=100, nb_epoch=10, threshold=0.5):
    words_size = 61681
    max_sentence_length = 788

    # 数据shuffle
    indices = permutation(X.shape[0])  # shape[0]表示第0轴的长度，通常是训练数据的数量
    X = X[indices]
    y = y[indices]
    # 分割数据
    X_train, X_test = X[:0.9*len(X)], X[0.9*len(X):]
    y_train, y_test = y[:0.9*len(y)], y[0.9*len(y):]
    X_train = sequence.pad_sequences(X_train, maxlen=max_sentence_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_sentence_length)

    print('Build model...')
    model = Sequential()
    model.add(Embedding(words_size+1, word_vec_len))
    model.add(LSTM(word_vec_len, 50))  # try using a GRU instead, for fun
    # model.add(GRU(word_vec_len, 50))
    model.add(Dropout(0.5))
    model.add(Dense(50, 1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adagrad', class_mode="binary")

    model.fit(X_train, y_train, shuffle=True, nb_epoch=nb_epoch, batch_size=batch_size, validation_split=0.1, show_accuracy=True)

    probas = model.predict_proba(X_test)
    classes = get_class(probas, threshold)
    accuracy = accuracy_score(y_test, classes)
    precision = precision_score(y_test, classes)
    recall = recall_score(y_test, classes)
    f1 = f1_score(y_test, classes)
    print '========阈值为%f时的结果========' % threshold
    print 'accuracy:', accuracy
    print 'precision:', precision
    print 'recall:', recall
    print 'F1:', f1

    return accuracy, precision, recall, f1


if __name__ == '__main__':
    # X, y = get_data('../Files/yahoo.data.dat')
    X, y = get_data('../Files/yahooAnswer.txt')
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for threshold in thresholds:
        accuracy, precision, recall, f1 = [], [], [], []
        for i in xrange(5):
            print 'Experiment', i+1
            acc, pre, rec, f = lstm(X, y, threshold=threshold)
            accuracy.append(acc)
            precision.append(pre)
            recall.append(rec)
            f1.append(f)
        print '========阈值为%f时的平均结果========' % threshold
        print 'accuracy:', np.mean(accuracy)
        print 'precision:', np.mean(precision)
        print 'recall:', np.mean(recall)
        print 'F1:', np.mean(f1)
