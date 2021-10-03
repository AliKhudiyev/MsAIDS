from tkinter import *
from object import *


class Renderer:
    def __init__(self, canvas):
        self.canvas = canvas
        self.width = 600 # canvas.winfo_width()
        self.height = 400 # canvas.winfo_height()
        self.objects = []
    
    def submit(self, objects):
        self.objects = objects

    def flush(self):
        self.canvas.delete('all')
        self.canvas.create_line(self.width/2, 0, self.width/2, self.height, dash=(2,2), fill='white')
        # print(self.width, self.height)
        for object_ in self.objects:
            if object_.type == Object.PLAYER:
                x0 = object_.x - object_.width / 2
                y0 = object_.y - object_.height / 2
                x1 = object_.x + object_.width / 2
                y1 = object_.y + object_.height / 2
                self.canvas.create_rectangle(x0, y0, x1, y1, fill='green')
            elif object_.type == Object.BALL:
                x0 = object_.x - object_.size / 2
                y0 = object_.y - object_.size / 2
                x1 = object_.x + object_.size / 2
                y1 = object_.y + object_.size / 2
                # print(x0,y0,x1,y1,object_.size)
                self.canvas.create_oval(x0, y0, x1, y1, fill='red', outline='red')
