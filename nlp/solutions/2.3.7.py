# PMI по формуле из лекции, логарифм натуральный (np.log)
# p(w,l) - вероятность того, что оба события совершились одновременно (и там и там единица).
# p(w) , p(l) - вероятности реализации случайных событий.

import sys
from io import StringIO
import numpy as np


# def parse_array(s):
#     return np.array([int(s.strip()) for s in s.strip().split(' ')])

# def read_array():
#     return parse_array(sys.stdin.readline())

def calculate_pmi(a, b):
    P_a = np.count_nonzero(a == 1) / a.shape[0]
    P_b = np.count_nonzero(b == 1) / b.shape[0]
    
    c = [i for  i, j in zip(a,b) if i == j == 1]       
    P_a_b = len(c) / a.shape[0]

    # print(c)
    # print(type(c))
    
    return np.log(P_a_b / (P_a * P_b))  # ваше решение

# a = read_array()
# b = read_array()

a = '''1 0 0 1 1 0 1'''
b = '''1 0 0 0 1 0 1'''
a = np.loadtxt(StringIO(a))
b = np.loadtxt(StringIO(b))

pmi_value = calculate_pmi(a, b)

print('{:.6f}'.format(pmi_value))


# Гениально!!!
# def calculate_pmi(a, b):
#     return np.log(np.mean(a*b)/np.mean(a)/np.mean(b))  # ваше решение