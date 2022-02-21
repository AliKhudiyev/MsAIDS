from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from buffer import *
from ctypes import sizeof


class Renderer:
    MAX_TRIANGLE = 1000

    def __init__(self, program):
        self.vao = Buffer()
        self.program = program
        self.renderables = []

        self.vertex_count = 0
        self.index_count = 0
        self.triangle_count = 0
        self.vertices = (ctypes.c_float * (3*Renderer.MAX_TRIANGLE))()
        self.indices = (ctypes.c_uint * (3*Renderer.MAX_TRIANGLE))()

        self.vao.layout([3])

    def add(self, renderable):
        self.renderables.append(renderable)

        tr_count = len(renderable.vertices) - 2
        index = self.vertex_count
        for i in range(0, tr_count, 3):
            self.indices[self.index_count+i] = self.vertex_count
            self.indices[self.index_count+i+1] = index + 1
            self.indices[self.index_count+i+2] = index + 2
            self.index_count += 3
            index += 1
        for vertex in renderable.vertices:
            for coord in vertex:
                self.vertices[self.vertex_count] = coord
                self.vertex_count += 1
        self.triangle_count += tr_count

    def remove(self, renderable):
        self.renderables.remove(renderable)

    def draw(self):
        glClearColor(0.0, 0.5, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)
        self.vao.enable()
        self.program.enable()

        self.vao.set_data(3*Renderer.MAX_TRIANGLE, self.vertices, self.indices)
        glDrawElements(GL_TRIANGLES, self.index_count*0+3, GL_UNSIGNED_INT, ctypes.c_void_p(0))
        glutSwapBuffers()

        self.program.disable()
        self.vao.disable()
        self.renderables = []
        self.vertex_count = 0
        self.index_count = 0
        self.triangle_count = 0

