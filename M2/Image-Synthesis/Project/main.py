from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import numpy as np
import random, time
import ctypes, sys

vao = GLuint(0)
vbo = GLuint(0)
null = ctypes.c_void_p(0)

vertices = np.array([
    0.0, 0.5, 0.0, 
    0.5, 0.0, 0.0, 
    -0.5, 0.0, 0.0], 
    dtype=np.float32)

program = None

strVertexShader = """
#version 330 core
layout (location = 0) in vec4 pos;
void main()
{
   gl_Position = vec4(pos.x, pos.y, pos.z, 1.0);
}
"""
strFragmentShader = """
#version 330 core
out vec4 outputColor;
void main()
{
   outputColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
}
"""
# 120

def handle_shaders():
    global program, strVertexShader, strFragmentShader

    vs = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vs, [strVertexShader])
    glCompileShader(vs)
    if not glGetShaderiv(vs, GL_COMPILE_STATUS):
        print('vertex shader compilation error')
        print(glGetShaderInfoLog(vs))

    fs = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fs, [strFragmentShader])
    glCompileShader(fs)
    if not glGetShaderiv(vs, GL_COMPILE_STATUS):
        print('fragment shader compilation error')
        print(glGetShaderInfoLog(vs))

    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)

def display():
    global vao
    global vbo, program

    print(vbo, program)

    glClear(GL_COLOR_BUFFER_BIT)

    glUseProgram(program)
    glBindVertexArray(vao)
    # glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glDrawArrays(GL_TRIANGLES, 0, 3)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    glutSwapBuffers()

def idle():
    pass

# Initialize GLUT
glutInit()
# Initialize the window with double buffering and RGB colors 
glutInitDisplayMode(GLUT_3_2_CORE_PROFILE, GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
# Set the window size to 500x500 pixels 
glutInitWindowSize(500, 500)
# Set the initial window position to (50, 50) 
glutInitWindowPosition(50, 50)
# Create the window and give it a title
glutCreateWindow("Ali Khudiyev")


vao = glGenVertexArrays(1)
# print('array object id:', vao)
vbo = glGenBuffers(1)
# print('buffer object id:', vbo)

glBindVertexArray(vao)
glBindBuffer(GL_ARRAY_BUFFER, vbo)

glBufferData(GL_ARRAY_BUFFER, 9*4, vertices, GL_STATIC_DRAW)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, null)
glEnableVertexAttribArray(0)
handle_shaders()

# a = 2
# print(sys.getsizeof(a))
# print(sys.getsizeof(null), ctypes.sizeof(null))


# Define display callback 
glutDisplayFunc(display)
# Define idle callback
glutIdleFunc(idle)
# Begin event loop
glutMainLoop()
