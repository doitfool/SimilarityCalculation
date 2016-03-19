# coding: utf-8
"""
    Project:    NLPSimilarity
    File:   Doc2Vec
    Author: AC
    Date:   2016/2/18 22:16
    Description:    使用gensim训练enwiki数据得到doc的向量化表示
"""
import logging
import os
import sys
import multiprocessing

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument

if __name__ == '__main__':

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)

    #   预处理wiki数据
    # input_file = '/home/trecqa/AC/enwiki-latest-pages-articles.xml.bz2'
    input_file = '/home/trecqa/AC/wiki.en.d2v_text'
    logger.info('process wiki data')
    documents = TaggedLineDocument(input_file)
    logger.info('finished processing wiki data')

    #   训练word2vec模型
    output_model = '/home/trecqa/AC/wiki.en.text.d2v_model'
    output_vector = '/home/trecqa/AC/wiki.en.text.d2v_vector'
    logger.info('train doc2vec model')
    model = Doc2Vec(documents, size=100, window=8, min_count=5, workers=multiprocessing.cpu_count())
    logger.info('finished training word2vec model')

    logger.info('saving doc2vec model')
    model.save(output_model)
    model.save_word2vec_format(output_vector, binary=False)