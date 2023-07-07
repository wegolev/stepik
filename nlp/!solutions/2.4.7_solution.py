from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np

corpus = [
    'Казнить нельзя, помиловать. Нельзя наказывать.',
    'Казнить, нельзя помиловать. Нельзя освободить.',
    'Нельзя не помиловать.',
    'Обязательно освободить.']

#Получаем счетчики слов
TF = CountVectorizer().fit_transform(corpus)

#Строим IDF. К сожалению, в этом задании нам нужно только vectorizer.idf_
#Для стандартных случаев на этой строке все вычисления и заканчиваются.
#Обычно  TFIDF = vectorizer.fit_transform(corpus)
vectorizer = TfidfVectorizer(smooth_idf=False, use_idf=True)
vectorizer.fit_transform(corpus)

## из IDF  в DF
word_doc_freq = 1/np.exp(vectorizer.idf_ - 1)

#TF нормируем и сглаживаем логарифмом (требование задания)
TFIDF = np.log(TF/TF.sum(axis=1)+1) / word_doc_freq 

#Масштабируем признаки
scaledTFIDF = StandardScaler().fit_transform(TFIDF)

#Домножаем на np.sqrt((4-1)/4) для перевода из DDOF(0) в DDOF(1) для 4 текстов
#(требование задания) 
scaledTFIDF *= np.sqrt(3/4)

#Вывод в порядке возрастания DF
for l in scaledTFIDF[:,np.argsort(word_doc_freq)]:
    print (" ".join([ "%.2f" % d for d in l]))