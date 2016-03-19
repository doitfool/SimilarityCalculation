# coding: utf-8

"""
    Project:    NLPSimilarity
    File:   Word2Vec
    Author: AC
    Date:   2016/2/18 22:16
    Description:    使用gensim训练enwiki数据得到word的向量化表示
"""

import logging
import os
import sys
import multiprocessing
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def process_enwiki(input_file, output_file):
    space = ' '
    i = 0
    output = open(output_file, 'w')
    wiki = WikiCorpus(input_file, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        output.write(space.join(text) + '\n')
        i += 1
        if i % 10000 == 0:
            logger.info('Saved ' + str(i) + ' articles')
    output.close()

if __name__ == '__main__':

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)

    #   预处理wiki数据
    input_file = '/home/trecqa/AC/enwiki-latest-pages-articles.xml.bz2'
    output_file = '/home/trecqa/AC/wiki.en.w2v_text'
    logger.info('process wiki data')
    process_enwiki(input_file, output_file)
    logger.info('finished processing wiki data')

    #   训练word2vec模型
    output_model = '/home/trecqa/AC/wiki.en.text.w2v_model'
    output_vector = '/home/trecqa/AC/wiki.en.text.w2v_vector'
    logger.info('train word2vec model')
    model = Word2Vec(LineSentence(output_file), size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())
    logger.info('finished training word2vec model')

    logger.info('saving word2vec model')
    model.save(output_model)
    model.save_word2vec_format(output_vector, binary=False)