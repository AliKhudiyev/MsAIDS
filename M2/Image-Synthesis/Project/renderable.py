from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


class Renderable:
    POINT = 0
    LINE = 1
    TRIANGLE = 2

    def __init__(self, vertices, colors):
        self.vertices = vertices
        self.colors = []
        self.data = []
        self.data_pattern = []

        if colors is None:
            self.colors = [(1, 1, 1, 1)] * len(vertices)
        elif len(colors) == 1:
            for i in range(len(vertices)):
                self.colors.append(colors[0])
        else:
            self.colors = colors

        if len(self.vertices) == 1:
            self.type = Renderable.POINT
        elif len(self.vertices) == 2:
            self.type = Renderable.LINE
        else:
            self.type = Renderable.TRIANGLE

    def apply_color(self, colors):
        pass

    def apply_texture(self, texture):
        pass

class Triangle(Renderable):
    def __init__(self, x1, y1, x2, y2, x3, y3, colors):
        super().__init__([[x1, y1, 0.0], [x2, y2, 0.0], [x3, y3, 0.0]], colors)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3
        self.colors = colors

class Rectangle(Renderable):
    def __init__(self, x, y, width, height, colors):
        super().__init__([[x, y, 0.0], [x+width, y, 0.0], 
            [x+width, y+height, 0.0], [x, y+height, 0.0]], colors)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.colors = colors

