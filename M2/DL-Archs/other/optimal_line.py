import numpy as np
import random
import matplotlib.pyplot as plt


xs = np.linspace(0, 10, 10)
ys = [3.8934*x-4.142117+random.uniform(-3, 3) for x in xs]

plt.scatter(xs, ys);

sum_x2 = 0
sum_xy = 0

for (x, y) in zip(xs, ys):
    sum_x2 += x**2
    sum_xy += x*y

b = (sum(ys)*sum_x2 - sum(xs)*sum_xy)/(len(xs)*sum_x2-sum(xs)**2)
a = (sum_xy - b*sum(xs))/sum_x2

print(f'y={a}x+{b}')
plt.plot(range(10), [a*x+b for x in range(10)]);
plt.show()
