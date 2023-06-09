import collections
import sys
import re
import numpy as np

def tokenize_text_simple_regex(txt, min_token_size=0): # min_token_size - мин размер токена!!!
    TOKEN_RE = re.compile(r'[\w\d]+')
    txt = txt.lower()
    all_tokens = TOKEN_RE.findall(txt)
    
    return [token for token in all_tokens if len(token) >= min_token_size]


def tokenize_corpus(texts, tokenizer=tokenize_text_simple_regex, **tokenizer_kwargs):
    return [tokenizer(text, **tokenizer_kwargs) for text in texts]


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


def main():
    # with open('2.4.5.txt') as f:
    #     data = f.readlines()
    data = open('2.4.5.txt', 'r')
    print(data)
    
    tokenized_data = tokenize_corpus(data)
    MAX_SIZE=1000000
    MAX_DF=1
    MIN_COUNT=0

    #print(*tokenized_data, sep='\n')

    return build_vocabulary(tokenized_data, MAX_SIZE, MAX_DF, MIN_COUNT)
    
if __name__ == '__main__':
    sys.exit(main())