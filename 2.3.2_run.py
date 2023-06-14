import re
import sys


# модифицируйте это регулярное выражение
TOKENIZE_RE = re.compile(r'[а-яА-ЯёЁ]+|-?\d*[.,]?\d+|\S', re.I)


def tokenize(txt):
    return TOKENIZE_RE.findall(txt)


for line in sys.stdin:
    print(' '.join(tokenize(line.strip().lower())))