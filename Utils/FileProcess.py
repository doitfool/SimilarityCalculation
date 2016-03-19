# coding: utf-8

"""
    Project:    LSTM4CQA
    File:   Text
    Author: AC
    Date:   2016/2/23 15:04
    Description: 预处理数据集，使其符合keras神经元输入格式
"""

import numpy as np
from keras.preprocessing.text import text_to_word_sequence


# 生成句子集合
def generate_sentences_set(data_file):
    sentences = set()
    with open(data_file, 'r') as fr:
        lines = fr.readlines()
        for i in xrange(len(lines)):
            splits = lines[i].split('\t')
            sentences.add(splits[0])
            sentences.add(splits[1])
    return sentences


# 持久化存储句子集合
def save_sentences_set(data_file, sentences_file):
    with open(sentences_file, 'w') as fw:
        sentences = generate_sentences_set(data_file)
        for sentence in sentences:
            fw.write(sentence+'\n')


# 生成单词集合
def generate_words_set(data_file):
    words = set()
    with open(data_file, 'r') as fr:
        lines = fr.readlines()
        for i in xrange(len(lines)):
            splits = lines[i].split('\t')
            text = splits[0] + ' ' + splits[1]
            for word in text_to_word_sequence(text):   # 模型训练时，文本数字化也要使用该函数text_to_word_sequence
                words.add(word)
    words = sorted(words)
    return words


# 持久化存储单词集合
def save_words_set(data_file, words_file):
    with open(words_file, 'w') as fw:
        words = generate_words_set(data_file)
        for word in words:
            fw.write(word+'\n')

# 索引化单词
def generate_w2i_i2w_dict(data_file):
    w2i = dict()
    i2w = dict()
    words = generate_words_set(data_file)
    for i in xrange(len(words)):
        w2i[words[i]] = i
        i2w[i] = words[i]
    return w2i, i2w


# 句子数字化
def sentence2sequence(sentence, w2i):
    sequence = []
    words = text_to_word_sequence(sentence)
    for word in words:
        if word in w2i:
            sequence.append(w2i[word])
    return sequence

def sentences2matrix(sentences, w2i):
    matirx = []
    for sentence in sentences:
        matirx.append(sentence2sequence(sentence, w2i))
    return matirx


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

if __name__ == '__main__':
    # data_file = '../Files/yahoo.data.dat'
    data_file = '../Files/yahooAnswer.txt'
    sentences_file = '../Files/sentences.dat'
    words_file = '../Files/words.dat'

    # output = pd.read_csv('C:\Users\AC\PycharmProjects\NLPSimilarity\Files\yahoo.data.dat', names=['Query1', 'Query2', 'label', 'ID'], sep='\t')
    # print type(output)
    # print output
    # output = output.as_matrix()
    # print type(output)
    # print output

    words = generate_words_set(data_file)
    print '共有%d个词' % len(words)     # 61681
    sentences = generate_sentences_set(data_file)
    print '最长的句子包含%d个单词' % max([len(text_to_word_sequence(sentence)) for sentence in sentences])        # 788
    X, y = get_data(data_file)
    print type(X), len(X), X.ndim, X[0], X[0][0], X[0][1]  #    <type 'numpy.ndarray'> 20563 2 [[5859, 5489, 502, 5805, 3429, 9129] [5580, 5923, 10200, 3429, 9132]] [5859, 5489, 502, 5805, 3429, 9129] [5580, 5923, 10200, 3429, 9132]
    print type(y), len(y), y.ndim, y[0]     #   <type 'numpy.ndarray'> 20563 1