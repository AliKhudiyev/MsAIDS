from tkinter import *
from matplotlib import pyplot as plt
import numpy as np


root = Tk()
root.title('Lagrange interpolation')

canvas = Canvas(root, width=400, height=400)
canvas.pack()
canvas.create_line(0, 200, 400, 200, width=2)
canvas.create_line(200, 0, 200, 400, width=2)

points = []
polynomial = []


def add_point(event):
    global points

    try:
        if [event.x-200, event.y-200] not in points:
            points.append([event.x-200, event.y-200])
    except:
        pass
    render()

def remove_point(event):
    global points

    try:
        for point in points:
            if np.abs(point[0]-(event.x-200)) < 4 and np.abs(point[1]-(event.y-200)) < 4:
                points.remove(point)
    except:
        pass
    render()

def render():
    global points, canvas

    all_points = []
    coef = 1
    for i in range(-200, 201):
        y = 0
        for j, point in enumerate(points):
            nom = 1
            denom = 1
            for m in range(len(points)):
                if j != m:
                    nom *= (i - points[m][0])
                    denom *= (point[0] - points[m][0])
            y += (nom / denom) * point[1]
        if np.abs(y) > 200:
            coef = 200 / np.abs(y)
        all_points.append([i, y])
    # print(coef)

    canvas.delete('all')
    canvas.create_line(0, 200, 400, 200, width=2)
    canvas.create_line(200, 0, 200, 400, width=2)

    for point in points:
        canvas.create_oval(point[0]+200-3, point[1]+200-3, point[0]+200+3, point[1]+200+3, fill='black')
    
    if len(points) > 1:
        for i, point in enumerate(all_points[:-1]):
            canvas.create_line(point[0]+200, point[1]+200, all_points[i+1][0]+200, all_points[i+1][1]+200, fill='black')


canvas.bind('<Button-1>', func=add_point)
canvas.bind('<Button-2>', func=remove_point)

root.mainloop()