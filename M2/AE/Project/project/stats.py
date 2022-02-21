import numpy as np

with open('losses.txt') as f:
    context = f.read().split('\n')

losses = [float(x[1:]) for x in context[:-1]]

print('mean:', np.mean(losses))
print('std:', np.std(losses))
print('best:', np.min(losses))