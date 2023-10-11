# import sympy

# x, y, w, b, c = sympy.symbols('x y w b c')
# res = sympy.diff(-y*sympy.log(1/(1+sympy.exp(-w*x-b)))-(1-y)*sympy.log(1-1/(1+sympy.exp(-w*x-b)) + c * (w ** 2 + b ** 2)), w)
# print(str(sympy.simplify(res)))

# sample_expr_str = str(sympy.simplify(res))
# sample_expr = sympy.parsing.sympy_parser.parse_expr(sample_expr_str)
# sample_value = sample_expr.evalf(subs=dict(x=0.5, y=1, w=4, b=1, c=1))
# print(sample_value)

Если у кого-то возникают проблемы с принятием решений после применения sympy,
то просто возьмите свой ответ из п.1 и добавьте к нему производную по w от c*w**2
x*(-y*exp(b + w*x) - y + exp(b + w*x))/(exp(b + w*x) + 1) + 2 *c * w
Только это решение принял степик!!! И на следующую задачу тоже это вариант решения нужно использовать.

# from sympy import symbols, exp, log, diff
# w,y,b,x, c = symbols('w y b x c')
# print(diff(-y*log(1/(1+exp(-w*x-b)))-(1-y)*log(1-1/(1+exp(-w*x-b)))+c*(w**2+b**2), w).simplify())
# Хотя этот вариант тоже фигурурет в ответах. Причем и след задачи тоже
