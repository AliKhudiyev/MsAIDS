from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import ctypes

null = ctypes.c_void_p(0)


class VertexBuffer:
    def __init__(self):
        self.vbo = glGenBuffers(1)
        self.enable()

    def enable(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

    def disable(self):
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def set_data(self, count, data):
        glBufferData(GL_ARRAY_BUFFER, count*ctypes.sizeof(GLfloat), data, GL_STATIC_DRAW)

class IndexBuffer:
    def __init__(self):
        self.ibo = glGenBuffers(1)
        self.enable()

    def enable(self):
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)

    def disable(self):
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def set_data(self, count, data):
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, count*ctypes.sizeof(GLuint), data, GL_STATIC_DRAW)

class ArrayBuffer:
    def __init__(self):
        self.vao = glGenVertexArrays(1)
        self.enable()

    def enable(self):
        glBindVertexArray(self.vao)

    def disable(self):
        glBindVertexArray(0)

class Buffer(ArrayBuffer):
    def __init__(self):
        super().__init__()
        self.vbo = VertexBuffer()
        self.ibo = IndexBuffer()
        self.index = 0

    def layout(self, pattern):
        offset = 0
        for n_elem in pattern:
            glVertexAttribPointer(self.index, n_elem, GL_FLOAT, GL_FALSE, 
                    sum(pattern)*ctypes.sizeof(GLfloat), ctypes.c_void_p(offset))
            glEnableVertexAttribArray(self.index)
            offset += n_elem
            self.index += 1

    def set_data(self, count, vertex_data, index_data):
        self.vbo.set_data(count, vertex_data)
        self.ibo.set_data(count, index_data)
