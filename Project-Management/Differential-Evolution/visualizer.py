import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# logs = []
# with open('evolution.log', 'r') as f:
#     n_generation = f.readline()

#     for i in range(int(n_generation)):
#         line = f.readline().split(';')
#         generation_log = []
#         for token in line:
#             input_ = token.split(',')
#             tmp = []
#             for value in input_:
#                 tmp.append(float(value))
#             generation_log.append(tmp)
#         logs.append(generation_log)

#     print(logs)
# f.close()

# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# ax.plot
# plt.show()

fig = plt.figure()
#creating a subplot 
ax1 = fig.add_subplot(111, projection='3d')

def ackley2D(x, y):
    return -20*np.e**(-0.2*((x**2+y**2)/2)**(1/2)) - np.e**((np.cos(2*np.pi*x)+np.cos(2*np.pi*y))/2) + 20 + np.e

def ackley2D_complete():
    X = []
    Y = []
    Z = []

    count = 50
    diff = 32 * 2 / count

    for x in range(count):
        for y in range(count):
            X.append(-32 + x * diff)
            Y.append(-32 + y * diff)
            Z.append(ackley2D(X[-1], Y[-1]))
            
    return X, Y, Z

def animate(i):
    data = open('stock.txt','r').read()
    lines = data.split('\n')
    xs = []
    ys = []
    zs = []
    X, Y, Z = ackley2D_complete()
   
    for line in lines:
        x, y, z = line.split(',') # Delimiter is comma    
        xs.append(float(x))
        ys.append(float(y))
        zs.append(float(z))
   
    
    ax1.clear()
    ax1.plot(X, Y, Z)
    ax1.plot(xs, ys, zs, '.')

    plt.xlabel('x')
    plt.ylabel('y')
    # plt.title('DEA in Ackley function')	
	
    
ani = animation.FuncAnimation(fig, animate, interval=1500) 
plt.show()
