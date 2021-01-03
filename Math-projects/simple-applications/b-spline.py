from tkinter import *
import numpy as np
from math import comb


root = Tk()
root.title('Bezier curve')

canvas = Canvas(root, width=400, height=400)
canvas.pack()
canvas.create_line(0, 200, 400, 200, width=2)
canvas.create_line(200, 0, 200, 400, width=2)

points = []
drag_index = -1
mouse_pos = [0, 0]
T = np.arange(0, 1, 0.01)

def add_point(event):
    global points, canvas

    try:
        if [event.x-200, event.y-200] not in points:
            if len(points) > 1:
                points.insert(len(points)-1, [event.x-200, event.y-200])
            else:
                points.append([event.x-200, event.y-200])
    except:
        pass
    drag_on = False
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

def drag_point(event):
    global points, drag_index
    
    if drag_index != -1:
        points[drag_index] = [event.x-200, event.y-200]
        render()

def drag_switch(label):
    global drag_index, mouse_pos

    # print(label.char)
    if label.char == 'd':
        if drag_index != -1:
            drag_index = -1
        else:
            for i, point in enumerate(points):
                if np.abs(point[0] - mouse_pos[0]) < 4 and np.abs(point[1]-mouse_pos[1]) < 4:
                    drag_index = i

def motion(event):
    global mouse_pos

    mouse_pos = [event.x-200, event.y-200]
    global points, drag_index
    
    if drag_index != -1:
        points[drag_index] = [event.x-200, event.y-200]
        render()

def render():
    global points, canvas, T

    all_points = []
    for t in T:
        x = 0
        y = 0
        for i, point in enumerate(points):
            coef = comb(len(points)-1, i) * (t**i) * (1-t)**(len(points)-1-i)
            x += point[0] * coef
            y += point[1] * coef
        all_points.append([x, y])
    all_points.append(points[-1])

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
canvas.bind('<Motion>', func=motion)
root.bind('<Key>', func=drag_switch)

root.mainloop()