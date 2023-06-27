import sys
import numpy as np
import scipy.sparse
import torch
from torch.utils.data import Dataset

tokenized_texts = [['казнить', 'нельзя', 'помиловать', 'нельзя', 'наказывать'], 
                   ['казнить', 'нельзя', 'помиловать', 'нельзя', 'освободить'], 
                   ['нельзя', 'не', 'помиловать'], 
                   ['обязательно', 'освободить']]
word2id = {'нельзя': 0, 'помиловать': 1, 'казнить': 2, 'освободить': 3, 
           'наказывать': 4, 'не': 5, 'обязательно': 6}
word2freq = np.array([0.75, 0.75, 0.5 , 0.5 , 0.25, 0.25, 0.25], dtype=float)

def vectorize_texts(tokenized_texts, word2id, word2freq, mode='tfidf', scale=True):
    assert mode in {'tfidf', 'idf', 'tf', 'bin'}

    # считаем количество употреблений каждого слова в каждом документе
    result = scipy.sparse.dok_matrix((len(tokenized_texts), len(word2id)), dtype='float32')
    for text_i, text in enumerate(tokenized_texts):
        for token in text:
            if token in word2id:
                result[text_i, word2id[token]] += 1

    # получаем бинарные вектора "встречается или нет"
    if mode == 'bin':
        result = (result > 0).astype('float32')

    # получаем вектора относительных частот слова в документе
    elif mode == 'tf':
        result = result.tocsr()
        result = result.multiply(1 / result.sum(1))

    # полностью убираем информацию о количестве употреблений слова в данном документе,
    # но оставляем информацию о частотности слова в корпусе в целом
    elif mode == 'idf':
        result = (result > 0).astype('float32').multiply(1 / word2freq)

    # учитываем всю информацию, которая у нас есть:
    # частоту слова в документе и частоту слова в корпусе
    elif mode == 'tfidf':
        result = result.tocsr()
        result = result.multiply(1 / result.sum(1))  # разделить каждую строку на её длину
        result = result.multiply(1 / word2freq)  # разделить каждый столбец на вес слова

    if scale:
        result = result.tocsc()
        result -= result.min()
        result /= (result.max() + 1e-6)

    return result.tocsr()

def main():

    return vectorize_texts(tokenized_texts, word2id, word2freq, mode='tfidf', scale=True)
    
if __name__ == '__main__':
    sys.exit(main())