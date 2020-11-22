import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

logs = []
counter = 0
dimension = 2

X = []
Y = []
Z = []

with open('build/evolution.log', 'r') as f:
    population_size = int(f.readline())
    raw_logs = f.readlines()
    
    generation_log = []
    for i, raw_log in enumerate(raw_logs):
        agent_log = []

        for coord in raw_log.split(','):
            agent_log.append(float(coord))

        dimension = len(agent_log)
        if dimension > 3:
            print(agent_log)
            print('Function has to be 2/3D!')
            exit()
        generation_log.append(agent_log)

        if i and i % (population_size - 1) == 0:
            logs.append(generation_log)
            generation_log = []
    # print(logs)
    # exit()
f.close()

with open('build/function.out', 'r') as f:
    raw_outs = f.readlines()
    
    for raw_out in raw_outs:
        out = []
        for raw_out_unit in raw_out.split(','):
            out.append(float(raw_out_unit))
        X.append(out[0])
        Y.append(out[1])
        Z.append(out[-1])
    # print(function_out[:10])
    # exit()
f.close()

fig = plt.figure()
#creating a subplot 
ax = None
if dimension == 2:
    ax = fig.add_subplot(111)
else:
    ax = fig.add_subplot(111, projection='3d')

def animate(i):
    global logs
    global counter
    global dimension

    global X
    global Y
    global Z

    global ax

    xs = []
    ys = []
    zs = []
    # X, Y, Z = ackley2D_complete()
    
    if counter >= len(logs):
        counter = len(logs) - 1
    
    generation_log = logs[counter]
    for agent_log in generation_log:
        xs.append(agent_log[0])
        ys.append(agent_log[1])
        zs.append(agent_log[-1])
    
    counter += 10
    
    ax.clear()

    if dimension == 2:
        ax.plot(X, Z)
        ax.plot(xs, zs, '.')
    else:
        ax.plot(X, Y, Z)
        ax.plot(xs, ys, zs, '.')

    plt.xlabel('x')
    plt.ylabel('y')
	
    
ani = animation.FuncAnimation(fig, animate, interval=500) 
plt.show()
