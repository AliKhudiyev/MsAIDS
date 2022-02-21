from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


class Shader:
    def __init__(self):
        self.program = glCreateProgram()
        self.vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        self.fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)

    def load_vertex(self, vertex_shader_filepath):
        vs = self.vertex_shader
        with open(vertex_shader_filepath, 'r') as f:
            buf = f.read()
            glShaderSource(vs, [buf])
            glCompileShader(vs)
            if not glGetShaderiv(vs, GL_COMPILE_STATUS):
                print('ERROR: vertex shader compilation failed!')
                print(glGetShaderInfoLog(vs))


    def load_fragment(self, fragment_shader_filepath):
        fs = self.fragment_shader
        with open(fragment_shader_filepath, 'r') as f:
            buf = f.read()
            glShaderSource(fs, [buf])
            glCompileShader(fs)
            if not glGetShaderiv(fs, GL_COMPILE_STATUS):
                print('ERROR: fragment shader compilation failed!')
                print(glGetShaderInfoLog(fs))

    def compile(self):
        glAttachShader(self.program, self.vertex_shader)
        glAttachShader(self.program, self.fragment_shader)
        glLinkProgram(self.program)
        if not glGetProgramiv(self.program, GL_LINK_STATUS):
            print('ERROR: linking program failed!')
            print(glGetProgramInfoLog(self.program))
        glDeleteShader(self.vertex_shader)
        glDeleteShader(self.fragment_shader)

    def enable(self):
        glUseProgram(self.program)

    def disable(self):
        glUseProgram(0)

