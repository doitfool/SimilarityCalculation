# coding: utf-8

"""
    Project:    LSTM4CQA
    File:   RNN_Graph
    Author: AC
    Date:   2016/2/24 20:52
    Description:    1 layer BLSTM_I
"""

import numpy as np
from numpy.random import permutation
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Reshape, Merge
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from Utils.FileProcess import generate_w2i_i2w_dict, sentence2sequence
from birnn import BiDirectionLSTM
from Utils.statistics import *
from LSTM import get_class


# 获取X, y
def get_data(data_file):
    X = []
    y = []
    w2i, i2w = generate_w2i_i2w_dict(data_file)
    with open(data_file, 'r') as fr:
        for line in fr:
            query1, query2, label = line.split('\t')[:3]
            query = (sentence2sequence(query1, w2i), sentence2sequence(query2, w2i))
            X.append(query)
            y.append(int(label))
    X = np.array(X)
    y = np.array(y)
    return X, y


def blstm_I(X, y, word_vec_len=300, batch_size=16, nb_epoch=10, threshold=0.5):
    words_size = 13033
    max_sentence_length = 29   # cut texts after this number of words (among top max_features most common words)


    # # 数据shuffle
    indices = permutation(X.shape[0])  # shape[0]表示第0轴的长度，通常是训练数据的数量
    X = X[indices]
    y = y[indices]

    X1 = [t[0] for t in X]
    X2 = [t[1] for t in X]
    X1 = np.array(X1)
    X2 = np.array(X2)

    X1_train, X1_test = X1[:0.9*len(X1)], X1[0.9*len(X1):]
    X2_train, X2_test = X2[:0.9*len(X2)], X2[0.9*len(X2):]
    y_train, y_test = y[:0.9*len(y)], y[0.9*len(y):]

    X1_train = sequence.pad_sequences(X1_train, maxlen=max_sentence_length)
    X1_test = sequence.pad_sequences(X1_test, maxlen=max_sentence_length)
    X2_train = sequence.pad_sequences(X2_train, maxlen=max_sentence_length)
    X2_test = sequence.pad_sequences(X2_test, maxlen=max_sentence_length)

    print('Build model...')

    left = Sequential()
    left.add(Embedding(words_size+1, word_vec_len))
    # left.add(BiDirectionLSTM(word_vec_len, 100, output_mode='concat', return_sequences=True))
    # left.add(BiDirectionLSTM(100*2, 100, output_mode='sum', return_sequences=True))
    left.add(BiDirectionLSTM(word_vec_len, 100, output_mode='sum', return_sequences=True))


    right = Sequential()
    right.add(Embedding(words_size+1, word_vec_len))
    # right.add(BiDirectionLSTM(word_vec_len, 100, output_mode='concat', return_sequences=True))
    # right.add(BiDirectionLSTM(100*2, 100, output_mode='sum', return_sequences=True))
    right.add(BiDirectionLSTM(word_vec_len, 100, output_mode='sum', return_sequences=True))

    model = Sequential()
    model.add(Merge([left, right], mode='sum'))     # 0.1.3 keras merge mode: one of {sum, concat}
                                                    # latest keras merge_mode: one of {sum, mul, concat, ave, join, cos, dot}

    model.add(Reshape(100 * max_sentence_length))
    model.add(BatchNormalization((100 * max_sentence_length,)))
    model.add(Dense(100 * max_sentence_length, 50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, 1, activation='sigmoid'))

    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='binary_crossentropy', optimizer=sgd, class_mode='binary')
    model.compile(loss='binary_crossentropy', optimizer='adagrad', class_mode='binary')
    print("Train...")
    model.fit([X1_train, X2_train], y_train, shuffle=True, nb_epoch=nb_epoch, batch_size=batch_size, validation_split=0.1, show_accuracy=True)

    probas = model.predict_proba([X1_test, X2_test])
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
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

def blstm_I_2layers(X, y, word_vec_len=300, batch_size=100, nb_epoch=10, threshold=0.5):
    words_size = 13033
    max_sentence_length = 29   # cut texts after this number of words (among top max_features most common words)

    # # 数据shuffle
    indices = permutation(X.shape[0])  # shape[0]表示第0轴的长度，通常是训练数据的数量
    X = X[indices]
    y = y[indices]

    X1 = [t[0] for t in X]
    X2 = [t[1] for t in X]
    X1 = np.array(X1)
    X2 = np.array(X2)

    X1_train, X1_test = X1[:0.9*len(X1)], X1[0.9*len(X1):]
    X2_train, X2_test = X2[:0.9*len(X2)], X2[0.9*len(X2):]
    y_train, y_test = y[:0.9*len(y)], y[0.9*len(y):]
	
    X1_train = sequence.pad_sequences(X1_train, maxlen=max_sentence_length)
    X1_test = sequence.pad_sequences(X1_test, maxlen=max_sentence_length)
    X2_train = sequence.pad_sequences(X2_train, maxlen=max_sentence_length)
    X2_test = sequence.pad_sequences(X2_test, maxlen=max_sentence_length)

    print('Build model...')

    left = Sequential()
    left.add(Embedding(words_size+1, word_vec_len))
    left.add(BiDirectionLSTM(word_vec_len, 100, output_mode='concat', return_sequences=True))
    left.add(BiDirectionLSTM(100*2, 100, output_mode='sum', return_sequences=True))
    # left.add(BiDirectionLSTM(word_vec_len, 100, output_mode='sum', return_sequences=True))


    right = Sequential()
    right.add(Embedding(words_size+1, word_vec_len))
    right.add(BiDirectionLSTM(word_vec_len, 100, output_mode='concat', return_sequences=True))
    right.add(BiDirectionLSTM(100*2, 100, output_mode='sum', return_sequences=True))
    # right.add(BiDirectionLSTM(word_vec_len, 100, output_mode='sum', return_sequences=True))

    model = Sequential()
    model.add(Merge([left, right], mode='sum'))     # 0.1.3 keras merge mode: one of {sum, concat}
                                                    # latest keras merge_mode: one of {sum, mul, concat, ave, join, cos, dot}

    model.add(Reshape(100 * max_sentence_length))
    model.add(BatchNormalization((100 * max_sentence_length,)))
    model.add(Dense(100 * max_sentence_length, 50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, 1, activation='sigmoid'))

    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='binary_crossentropy', optimizer=sgd, class_mode='binary')
    model.compile(loss='binary_crossentropy', optimizer='adagrad', class_mode='binary')
    print("Train...")
    model.fit([X1_train, X2_train], y_train, shuffle=True, nb_epoch=nb_epoch, batch_size=batch_size, validation_split=0.1, show_accuracy=True)

    probas = model.predict_proba([X1_test, X2_test])
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
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    X, y = get_data('../Files/yahoo.data.dat')
    blstm_I(X, y)