#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/27 3:13
# @Author  : glq
# @Email   : wycmglq@outlook.com
# @File    : 04get_word_emb.py
# @Do      : something

import gensim
from gensim.models.word2vec import Word2Vec
if __name__ == '__main__':
    #训练结束了，所以底下train_word2vec不需要调用了
    #generate_word_txt()
    # model = train_word2vec()
    w2v_model = gensim.models.Word2Vec.load('../../data/middle_data/word2vec_model.w2v')  # 调用模型
    # w2v_model.wv.save_word2vec_format("../../data/middle_result/word2vec.txt",binary=False) #将词向量保存为 txt格式
    #读取txt用下面的代码
    # f1 = open("./temp/word2vec.txt", 'r', encoding='utf-8')
    # vectors = f1.readlines()
    # f1.close()
    sim_words = w2v_model.most_similar(positive=['7005245'])
    for word, similarity in sim_words:
        print(word, similarity)  # 输出’女人‘相近的词语和概率
    print(w2v_model['7005245'] ) #输出某个单词的词向量
    # embedding_test = get_word_embedding('7005245')
    # print(embedding_test)