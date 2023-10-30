# реализация алгоритма взвешивания признаков  с помощью точечной взаимной информации (PMI) на python
# Вот пример реализации алгоритма взвешивания признаков с использованием точечной взаимной информации (PMI) на языке Python:

import math
from collections import defaultdict

def calculate_pmi(feature_matrix):
    num_documents = len(feature_matrix)
    feature_counts = defaultdict(int)
    cooccurrence_counts = defaultdict(int)

    # Подсчет количества документов, содержащих каждый признак,
    # а также подсчет взаимного вхождения пар признаков
    for document in feature_matrix:
        for feature in document:
            feature_counts[feature] += 1
            for other_feature in document:
                if other_feature != feature:
                    cooccurrence_counts[(feature, other_feature)] += 1
    
    print(feature_counts)
    print(cooccurrence_counts)

    # Расчет PMI для каждой пары признаков
    pmi_scores = defaultdict(float)
    for (feature, other_feature), cooccurrence_count in cooccurrence_counts.items():
        pmi = math.log((cooccurrence_count * num_documents) / (feature_counts[feature] * feature_counts[other_feature]))
        pmi_scores[(feature, other_feature)] = pmi

    return pmi_scores

# Пример использования
feature_matrix = [
    ['apple', 'banana', 'orange'],
    ['apple', 'orange'],
    ['banana', 'orange'],
    ['apple', 'banana']
]

pmi_scores = calculate_pmi(feature_matrix)

# Вывод PMI-оценок для каждой пары признаков
for (feature, other_feature), pmi in pmi_scores.items():
    print(f"PMI({feature}, {other_feature}) = {pmi}")
    
# Этот код реализует алгоритм взвешивания признаков с использованием точечной взаимной информации (PMI) на основе матрицы признаков feature_matrix. Алгоритм сначала подсчитывает количество документов, содержащих каждый признак, а также количество взаимного вхождения пар признаков. Затем он вычисляет PMI для каждой пары признаков и возвращает словарь с PMI-оценками.

# Приведенный пример иллюстрирует использование алгоритма на простом наборе данных. Вы можете изменить feature_matrix на свои данные для выполнения алгоритма на своем наборе признаков.