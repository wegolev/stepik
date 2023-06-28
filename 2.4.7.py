import sys
import numpy as np
import scipy.sparse

tokenized_texts = [['казнить', 'нельзя', 'помиловать', 'нельзя', 'наказывать'], 
                   ['казнить', 'нельзя', 'помиловать', 'нельзя', 'освободить'], 
                   ['нельзя', 'не', 'помиловать'], 
                   ['обязательно', 'освободить']]
word2id = {'наказывать': 4, 'не': 5, 'обязательно': 6, 'казнить': 2, 
           'освободить': 3, 'нельзя': 0, 'помиловать': 1}
word2freq = np.array([0.25, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75], dtype=float)

def vectorize_texts(tokenized_texts, word2id, word2freq, mode='tfidf', scale=True):
    assert mode in {'ltfidf', 'tfidf', 'idf', 'tf', 'bin'}

    # считаем количество употреблений каждого слова в каждом документе
    result = scipy.sparse.dok_matrix((len(tokenized_texts), len(word2id)), dtype='float32')
    for text_i, text in enumerate(tokenized_texts):
        for token in text:
            if token in word2id:
                result[text_i, word2id[token]] += 1
    print(result.tocsr())

    if mode == 'ltfidf':
        result = result.tocsr()
        result = result.multiply(1 / result.sum(1))  # разделить каждую строку на её длину
        result = result.log1p() # Return the natural logarithm of one plus the input array, element-wise. Calculates log(1 + x).
        result = result.multiply(1 / word2freq)  # разделить каждый столбец на вес слова

        print(result)

    if scale:
        result = result.tocsc()
        result -= result.mean(0)
        result /= result.std(0, ddof=1)

    return result.tocsr()

def main():
    return vectorize_texts(tokenized_texts, word2id, word2freq, mode='ltfidf', scale=True)
    
if __name__ == '__main__':
    sys.exit(main())