import sympy

x, y, w, b = sympy.symbols('x y w b')
res = sympy.diff(-y*sympy.log(1/(1+sympy.exp(-w*x-b)))-(1-y)*sympy.log(1-1/(1+sympy.exp(-w*x-b))), b)
print(str(sympy.simplify(res)))

#import sympy.parsing.sympy_parser

#sample_expr_str = '(-y*exp(b + w*x) - y + exp(b + w*x))/(exp(b + w*x) + 1)'
sample_expr_str = str(sympy.simplify(res))
sample_expr = sympy.parsing.sympy_parser.parse_expr(sample_expr_str)
sample_value = sample_expr.evalf(subs=dict(x=0.5, y=1, w=4, b=1))
print(sample_value)