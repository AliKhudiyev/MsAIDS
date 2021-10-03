from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
import time
from algorithm import *


graph = nx.Graph()
towns_df = pd.read_csv('towns.csv', sep=';')
roads_df = pd.read_csv('roads.csv', sep=';')

cities = towns_df['name'].tolist()
roads = []
roads_between = []
algorithms = ['Depth First', 'Breadth First', 'Greedy', 'A*', 'Uniform Cost']
heuristics = ['Distance', 'Time']

# print(roads_df.loc[(roads_df['town1'] == 1) & (roads_df['town2'] == 38)].values)

# Initialization of graph
for city in cities:
    graph.add_node(city)

for i, city1 in enumerate(cities):
    for j, city2 in enumerate(cities):
        if i == j:
            continue

        attributes = roads_df.loc[(roads_df['town1'] == i+1) & (roads_df['town2'] == j+1)].values
        if len(attributes) == 0:
            continue
        
        graph.add_edge(city1, city2, distance=attributes[0][2], time=attributes[0][3])
# = = = = = = = = = = =

# nx.draw(graph)
# nx.draw_networkx(graph)
# plt.show()

# for neighbour in graph.neighbors('Laon'):
#     print(neighbour, graph.get_edge_data('Laon', neighbour))
#     print(neighbour, graph.get_edge_data(neighbour, 'Laon'))

# depth_first_search(graph, 'Laon', 'Melun', 'distance')

# = = = = = GUI = = = = = = =
root = Tk()
root.title('Route Planner')

max_width = 1000
max_height = 650

print(max_width, max_height)
# exit(1)

canvas = Canvas(root)
canvas.pack()

frame = Frame(root)
frame.pack()

label_distance = Label(frame, text='Distance(km): -')
label_time = Label(frame, text='Time(min): -')
label_computationTime = Label(frame, text='Computation time(ms): -')
label_exploredNodes = Label(frame, text='Explored nodes: -')

def run_algorithm(algorithm, G, start, end, heuristic):
    # print(f'runnig {algorithm} with {start} and {end} with heuristic of {heuristic}')
    start_time = time.time()
    path = find_path(algorithm, G, start, end, heuristic)
    computation_time = time.time() - start_time

    total_distance = 0
    total_time = 0

    for i in range(len(path)-1):
        total_distance += G.get_edge_data(path[i], path[i+1])['distance']
        total_time += G.get_edge_data(path[i], path[i+1])['time']

    label_distance.config(text=f'Distance(km): {total_distance}')
    label_time.config(text=f'Time(min): {total_time}')
    label_computationTime.config(text=f'Computation time(ms): {np.round(computation_time * 1000, 2)}')
    label_exploredNodes.config(text=f'Explored nodes: {len(path)}')

    return path

image = Image.open('France_map_admin_1066_1024.png')
width, height = image.width, image.height

if height > max_height:
    height = max_height
if width > max_width:
    width = max_width

image = image.resize((width, height), Image.ANTIALIAS)
map = ImageTk.PhotoImage(image)
canvas.config(width=width, height=height)
canvas.create_image(0, 0, anchor=NW, image=map)

Label(frame, text='Starting city').grid(row=0, column=0)
Label(frame, text='Ending city').grid(row=0, column=2)
Label(frame, text='Algorith').grid(row=1, column=0)
Label(frame, text='Heuristic').grid(row=1, column=2)

label_distance.grid(row=2, column=0)
label_time.grid(row=2, column=1)
label_computationTime.grid(row=2, column=2)
label_exploredNodes.grid(row=2, column=3)

var_startingCity = StringVar(frame)
var_startingCity.set('') # default value

var_endingCity = StringVar(frame)
var_endingCity.set('') # default value

var_algorithm = StringVar(frame)
var_algorithm.set('Greedy') # default value

var_heuristic = StringVar(frame)
var_heuristic.set('Distance') # default value

OptionMenu(frame, var_startingCity, *cities).grid(row=0, column=1)
OptionMenu(frame, var_endingCity, *cities).grid(row=0, column=3)
OptionMenu(frame, var_algorithm, *algorithms).grid(row=1, column=1)
OptionMenu(frame, var_heuristic, *heuristics).grid(row=1, column=3)

button_run = Button(frame, text='Run', command=lambda: display_path(
        run_algorithm(var_algorithm.get().lower(), graph, var_startingCity.get(), var_endingCity.get(), var_heuristic.get().lower())
        )
    )
Button(frame, text='Quit', command=root.destroy).grid(row=3, column=2, columnspan=2)
button_run.grid(row=3, column=0, columnspan=2)

# Displaying cities and roads in the map
def display_path(cities):
    for i in range(len(roads_between)):
        canvas.itemconfig(roads[i], fill='green', width=2)
    
    for i in range(len(cities)-1):
        for j in range(len(roads_between)):
            if (roads_between[j][0] == cities[i] and roads_between[j][1] == cities[i+1]) or \
                (roads_between[j][1] == cities[i] and roads_between[j][0] == cities[i+1]):
                canvas.itemconfig(roads[j], fill='red', width=4)


def update_map(color='green', width=2):
    print(f'updating map with color {color}')

    for i, city1 in enumerate(cities):
        id1 = towns_df.loc[towns_df['name'] == city1]['id'].values[0]
        lat1 = towns_df.loc[towns_df['name'] == city1]['latitude'].values[0]
        lon1 = towns_df.loc[towns_df['name'] == city1]['longitude'].values[0]

        a1 = map.height() / (41 - 51.5)
        b1 = -a1 * 51.5
        a2 = map.width() / (10 + 5.8)
        b2 = a2 * 5.8

        x1 = a2 * lon1 + b2
        y1 = a1 * lat1 + b1

        for j, city2 in enumerate(cities):
            id2 = towns_df.loc[towns_df['name'] == city2]['id'].values[0]
            if j <= i or len(roads_df.loc[(roads_df['town1'] == id1) & (roads_df['town2'] == id2)].values) == 0:
                continue
            
            lat2 = towns_df.loc[towns_df['name'] == city2]['latitude'].values[0]
            lon2 = towns_df.loc[towns_df['name'] == city2]['longitude'].values[0]

            a1 = map.height() / (41 - 51.5)
            b1 = -a1 * 51.5
            a2 = map.width() / (10 + 5.8)
            b2 = a2 * 5.8

            x2 = a2 * lon2 + b2
            y2 = a1 * lat2 + b1

            roads_between.append([city1, city2])
            roads.append(canvas.create_line(x1, y1, x2, y2, fill=color, width=width))
        canvas.create_oval(x1-2, y1-2, x1+2, y1+2, fill='red')

update_map()
root.mainloop()
# = = = = = = = = = = = = = =
