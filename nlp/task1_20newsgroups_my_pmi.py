import numpy as np
import scipy.sparse
import collections
from sklearn.datasets import fetch_20newsgroups
from dlnlputils_my.data import tokenize_corpus

train_source = fetch_20newsgroups(subset='train')
test_source = fetch_20newsgroups(subset='test')

# print('Количество обучающих текстов', len(train_source['data']))
# print('Количество тестовых текстов', len(test_source['data']))
# print()
# print(train_source['data'][0].strip())
# print()
# print('Метка', train_source['target'][0])
# print(50 * '=')

train_tokenized = tokenize_corpus(train_source['data'][:10])
test_tokenized = tokenize_corpus(test_source['data'][:2])
print('Длина train_tokenized', len(train_tokenized))
# print('train_tokenized as text:')
# print(' '.join(train_tokenized[0]))
print('train_tokenized:')
print(train_tokenized[:1])
print(50 * '=')

def build_vocabulary(tokenized_texts, max_size=1000000, max_doc_freq=0.8, min_count=5, pad_word=None):
    word_counts = collections.defaultdict(int)
    doc_n = 0

    # посчитать количество документов, в которых употребляется каждое слово
    # а также общее количество документов
    for txt in tokenized_texts:
        doc_n += 1
        unique_text_tokens = set(txt)
        for token in unique_text_tokens:
            word_counts[token] += 1

    # убрать слишком редкие и слишком частые слова
    word_counts = {word: cnt for word, cnt in word_counts.items()
                   if cnt >= min_count and cnt / doc_n <= max_doc_freq}

    # отсортировать слова по убыванию частоты
    sorted_word_counts = sorted(word_counts.items(),
                                reverse=True,
                                key=lambda pair: pair[1])

    # добавим несуществующее слово с индексом 0 для удобства пакетной обработки
    if pad_word is not None:
        sorted_word_counts = [(pad_word, 0)] + sorted_word_counts

    # если у нас по прежнему слишком много слов, оставить только max_size самых частотных
    if len(word_counts) > max_size:
        sorted_word_counts = sorted_word_counts[:max_size]

    # нумеруем слова
    word2id = {word: i for i, (word, _) in enumerate(sorted_word_counts)}

    # нормируем частоты слов
    word2freq = np.array([cnt / doc_n for _, cnt in sorted_word_counts], dtype='float32')

    return word2id, word2freq

def vectorize_texts(tokenized_texts, word2id, word2freq, mode='tfidf', scale=True):
    assert mode in {'pmi', 'log(tf+1)idf', 'tfidf', 'idf', 'tf', 'bin'}

    # считаем количество употреблений каждого слова в каждом документе
    result = scipy.sparse.dok_matrix((len(tokenized_texts), len(word2id)), dtype='float32')
    for text_i, text in enumerate(tokenized_texts):
        for token in text:
            if token in word2id:
                result[text_i, word2id[token]] += 1

    print(result.shape)
    print(result.todense())

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
    
    elif mode == 'log(tf+1)idf':
        result = result.tocsr()
        result = result.multiply(np.log1p(1 / result.sum(1) + 1))  # ln(TF+1) 
        result = result.multiply(1 / word2freq)  # разделить каждый столбец на вес слова

    elif mode == 'pmi':
        result = result.tocsr()
        result = result.multiply(1 / result.sum(1))   

    if scale:
        result = result.tocsc()
        result -= result.min()
        result /= (result.max() + 1e-6)
    
    return result.tocsr()

MAX_DF = 0.8
MIN_COUNT = 5
vocabulary, word_doc_freq = build_vocabulary(train_tokenized, max_doc_freq=MAX_DF, min_count=MIN_COUNT)
UNIQUE_WORDS_N = len(vocabulary)
print('Количество уникальных токенов', UNIQUE_WORDS_N)
print(list(vocabulary.items())[:10])
print(len(word_doc_freq))
print(word_doc_freq)
print(50 * '=')

VECTORIZATION_MODE = 'pmi'
train_vectors = vectorize_texts(train_tokenized, vocabulary, word_doc_freq, mode=VECTORIZATION_MODE)
test_vectors = vectorize_texts(test_tokenized, vocabulary, word_doc_freq, mode=VECTORIZATION_MODE)

# print('Размерность матрицы признаков обучающей выборки', train_vectors.shape)
# print('Размерность матрицы признаков тестовой выборки', test_vectors.shape)
# print()
# print('Количество ненулевых элементов в обучающей выборке', train_vectors.nnz)
# print('Процент заполненности матрицы признаков {:.2f}%'.format(train_vectors.nnz * 100 / (train_vectors.shape[0] * train_vectors.shape[1])))
# print()
# print('Количество ненулевых элементов в тестовой выборке', test_vectors.nnz)
# print('Процент заполненности матрицы признаков {:.2f}%'.format(test_vectors.nnz * 100 / (test_vectors.shape[0] * test_vectors.shape[1])))
