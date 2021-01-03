import numpy as np

vars = list()
n_var = 1

while True:
    var = input(f'Term {n_var}: ')
    if len(var) == 0:
        break
    if var not in vars:
        vars.append(var)
        n_var += 1

coefs_str = input('Enter coefficients for the terms of your polynomial: ').split(' ')
coefs_divs_str = list()

n_div = 1
while True:
    coefs_div_str = input(f'Enter coefficients for the terms of divisor {n_div}: ').split(' ')
    if coefs_div_str[0] == '':
        break
    if coefs_div_str not in coefs_divs_str:
        while len(coefs_div_str) < len(vars):
            coefs_div_str.append('0')
        coefs_divs_str.append(coefs_div_str)
        n_div += 1

while len(coefs_str) < len(vars):
    coefs_str.append('0')

print('= = = = = = = = = = = = = = = =')
poly = ''
for i, coef in enumerate(coefs_str):
    if float(coef) != 0:
        poly += f'({coef})*({vars[i]})'
        if i < len(coefs_str) - 1 and float(coefs_str[i+1]) != 0:
            poly += ' + '
print(f'P = {poly}')

for i, coefs in enumerate(coefs_divs_str):
    poly = ''
    for j, coef in enumerate(coefs):
        if float(coef) != 0:
            poly += f'({coef})*({vars[j]})'
            if j < len(coefs) - 1 and float(coefs[j+1]) != 0:
                poly += ' + '
    print(f'P[{i}] = {poly}')

coefs = [float(coef_str) for coef_str in coefs_str]
coefs_divs = [[float(coef_div_str) for coef_div_str in coefs_div_str] for coefs_div_str in coefs_divs_str]
quotients = [0 for i in range(len(coefs_divs_str))]
remainder = coefs

for i, coefs_div in enumerate(coefs_divs):
    for j, coef_div in enumerate(coefs_div):
        if coef_div != 0:
            quotient = coefs[j] / coef_div
            tmp = [quotient * c for c in coefs_div]
            coefs = [c1 - c2 for c1, c2 in zip(coefs, tmp)]
            quotients[i] = quotient
            remainder = coefs
            break

print(remainder)
print(f'\nP = ', end='')
for i, coefs in enumerate(coefs_divs_str):
    print(f'{np.around(quotients[i], 3)}*P[{i}]', end=' + ')

no_remainder = True
print('[', end='')
for i, r in enumerate(remainder):
    if r != 0:
        print(f'({r})*({vars[i]})', end='')
        no_remainder = False
        if i < len(remainder) - 1 and remainder[i+1] != 0:
            print(' + ', end='')
if no_remainder:
    print(0, end='')
print(']')
