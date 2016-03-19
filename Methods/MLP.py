# coding: utf-8

"""
    Project:    NLPSimilarity
    File:   MLP
    Author: AC
    Date:   2016/2/21 10:52
    Description:    
"""

import numpy as np
from numpy.random import permutation
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.normalization import BatchNormalization

from birnn import Transform
from Utils.FileProcess import generate_w2i_i2w_dict, sentence2sequence


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
    # print type(X), X.shape, X.size, X       # <type 'numpy.ndarray'> (20563L,)
    # print max([len(x) for x in X])          # 最大长度为 47
    # print type(y), y.shape, y.size, y       # <type 'numpy.ndarray'> (20563L,)
    return X, y


def mlp():
    words_size = 13033
    word_vec_len = 100
    batch_size = 100
    max_sentence_length = 47   # cut texts after this number of words (among top max_features most common words)
    X, y = get_data('../Files/yahoo.data.dat')

    # 数据shuffle
    indices = permutation(X.shape[0])  # shape[0]表示第0轴的长度，通常是训练数据的数量
    X = X[indices]
    y = y[indices]

    X_train, X_test = X[:0.8*len(X)], X[0.8*len(X):]
    y_train, y_test = y[:0.8*len(y)], y[0.8*len(y):]

    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    print("Pad sequences (samples x time)")
    X_train = sequence.pad_sequences(X_train, maxlen=max_sentence_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_sentence_length)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Build model...')
    model = Sequential()
    model.add(Embedding(words_size, word_vec_len))  # (nb_samples, sequence_length, output_dim)

    model.add(Transform((word_vec_len,)))   # transform from 3d dimensional input to 2d input for mlp
    model.add(Dense(input_dim=word_vec_len, output_dim=25, init='uniform', activation='tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(input_dim=25, output_dim=15, init='uniform', activation='tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(input_dim=15, output_dim=8, init='uniform', activation='tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(input_dim=8, output_dim=1, init='uniform', activation='sigmoid'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, class_mode='binary')
    # model.compile(loss='mean_squared_error', optimizer=sgd, class_mode='binary')


    print("Train...")
    model.fit(X_train, y_train, shuffle=True, batch_size=batch_size, nb_epoch=5, validation_split=0.1, show_accuracy=True)

    score = model.evaluate(X_test, y_test, batch_size=10)
    print('Test score:', score)

    classes = model.predict_classes(X_test, batch_size=batch_size)
    acc = np_utils.accuracy(classes, y_test)
    print('Test accuracy:', acc)    # ('Test accuracy:', 0.64478482859226838)
    #
    # json_mlp = model.to_json()
    # open('../Files/mlp.json', 'w').write(json_mlp)
    # model.save_weights('../Files/mlp_weights.h5')

if __name__ == '__main__':
    mlp()