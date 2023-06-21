# PMI по формуле из лекции, логарифм натуральный (np.log)
# p(w,l) - вероятность того, что оба события совершились одновременно (и там и там единица).
# p(w) , p(l) - вероятности реализации случайных событий.

import sys
import numpy as np


def parse_array(s):
    return np.array([int(s.strip()) for s in s.strip().split(' ')])

def read_array():
    return parse_array(sys.stdin.readline())

def calculate_pmi(a, b):
    P_a = np.count_nonzero(a == 1) / a.shape[0]
    P_b = np.count_nonzero(b == 1) / b.shape[0]
    P_a_b = np.intersect1d(a, b).shape[0] / a.shape[0]
    
    return np.log(P_a_b / (P_a * P_b))  # ваше решение

a = read_array()
b = read_array()
pmi_value = calculate_pmi(a, b)

print('{:.6f}'.format(pmi_value))