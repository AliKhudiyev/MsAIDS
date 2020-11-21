import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

logs = []
counter = 0

with open('build/evolution.log', 'r') as f:
    population_size = int(f.readline())
    raw_logs = f.readlines()
    print(population_size)
    generation_log = []
    for i, raw_log in enumerate(raw_logs):
        agent_log = []

        for coord in raw_log.split(','):
            agent_log.append(float(coord))
        generation_log.append(agent_log)

        if i and i % (population_size - 1) == 0:
            logs.append(generation_log)
            generation_log = []
    # print(logs)
    # exit()
f.close()

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
    global logs
    global counter

    xs = []
    ys = []
    zs = []
    X, Y, Z = ackley2D_complete()
    
    if counter >= len(logs):
        counter = len(logs) - 1
    
    generation_log = logs[counter]
    for agent_log in generation_log:
        xs.append(agent_log[0])
        ys.append(agent_log[1])
        zs.append(agent_log[2])
    
    counter += 10
    
    ax1.clear()
    ax1.plot(X, Y, Z)
    ax1.plot(xs, ys, zs, '.')

    plt.xlabel('x')
    plt.ylabel('y')
    # plt.title('DEA in Ackley function')	
	
    
ani = animation.FuncAnimation(fig, animate, interval=500) 
plt.show()
