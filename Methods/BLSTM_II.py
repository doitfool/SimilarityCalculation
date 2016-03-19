# coding: utf-8

"""
    Project:    LSTM4CQA
    File:   RNN
    Author: AC
    Date:   2016/2/24 15:36
    Description:    1 layer BLSTM_II
"""
import numpy as np
from numpy.random import permutation
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.convolutional import MaxPooling1D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.normalization import BatchNormalization

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
            query = (sentence2sequence(query1, w2i)+sentence2sequence(query2, w2i))
            X.append(query)
            y.append(int(label))
    X = np.array(X)
    y = np.array(y)
    return X, y


# 训练并存储模型
def blstm_II(X, y, word_vec_len=300, batch_size=100, nb_epoch=10, threshold=0.5):
    words_size = 13033
    max_sentence_length = 47   # cut texts after this number of words (among top max_features most common words)

    # 数据shuffle
    indices = permutation(X.shape[0])  # shape[0]表示第0轴的长度，通常是训练数据的数量
    X = X[indices]
    y = y[indices]

    X_train, X_test = X[:0.9*len(X)], X[0.9*len(X):]
    y_train, y_test = y[:0.9*len(y)], y[0.9*len(y):]
	
    X_train = sequence.pad_sequences(X_train, maxlen=max_sentence_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_sentence_length)

    print('Build model...')
    model = Sequential()

    model.add(Embedding(words_size+1, word_vec_len))
    # Stacked up BiDirectionLSTM layers
    model.add(BiDirectionLSTM(word_vec_len, 100, output_mode='sum', return_sequences=True))
    # model.add(BiDirectionLSTM(50, 50, output_mode='sum', return_sequences=True))

    # MLP layers
    model.add(Reshape(100 * max_sentence_length, ))
    model.add(BatchNormalization((100 * max_sentence_length,)))
    model.add(Dense(100 * max_sentence_length, 50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, 1, activation='sigmoid'))

    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='binary_crossentropy', optimizer=sgd, class_mode='binary')
    model.compile(loss='binary_crossentropy', optimizer='adagrad', class_mode='binary')

    print("Train...")
    model.fit(X_train, y_train, shuffle=True, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.1, show_accuracy=True)

    probas = model.predict_proba(X_test)
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

    # 存储模型和参数
    # blstm_json = model.to_json()
    # open('BLSTM_model_architecture.json', 'w').write(blstm_json)
    # model.save_weights('../Files/BLSTM_model_weights.h5')


def blstm_II_2layers(X, y, word_vec_len=300, batch_size=100, nb_epoch=10, threshold=0.5):
    words_size = 13033
    max_sentence_length = 47   # cut texts after this number of words (among top max_features most common words)

    # 数据shuffle
    indices = permutation(X.shape[0])  # shape[0]表示第0轴的长度，通常是训练数据的数量
    X = X[indices]
    y = y[indices]

    X_train, X_test = X[:0.9*len(X)], X[0.9*len(X):]
    y_train, y_test = y[:0.9*len(y)], y[0.9*len(y):]
    print len(X_train), 'train sequences'
    print len(X_test), 'test sequences'

    print "Pad sequences (samples x time)"
    X_train = sequence.pad_sequences(X_train, maxlen=max_sentence_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_sentence_length)
    print 'X_train shape:', X_train.shape
    print 'X_test shape:', X_test.shape
    print 'y_train shape:', y_train.shape
    print 'y_test shape:', y_test.shape

    print('Build model...')
    model = Sequential()

    model.add(Embedding(words_size+1, word_vec_len))
    # Stacked up BiDirectionLSTM layers
    model.add(BiDirectionLSTM(word_vec_len, 100, output_mode='sum', return_sequences=True))
    model.add(BiDirectionLSTM(100, 100, output_mode='sum', return_sequences=True))

    # MLP layers
    model.add(Reshape(100 * max_sentence_length, ))
    model.add(BatchNormalization((100 * max_sentence_length,)))
    model.add(Dense(100 * max_sentence_length, 50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, 1, activation='sigmoid'))

    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='binary_crossentropy', optimizer=sgd, class_mode='binary')
    model.compile(loss='binary_crossentropy', optimizer='adagrad', class_mode='binary')

    print("Train...")
    model.fit(X_train, y_train, shuffle=True, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.1, show_accuracy=True)

    probas = model.predict_proba(X_test)
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

if __name__ == '__main__':
    X, y = get_data('../Files/yahoo.data.dat')
    blstm_II(X, y)