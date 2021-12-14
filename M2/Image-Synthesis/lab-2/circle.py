# Importing the necessary Modules
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import numpy as np
import random


center = (0.25, -0.5)
radius = 0.5


# Disply callback function
def display():
    # Reset background
    glClear(GL_COLOR_BUFFER_BIT)

    # Render scene
    render_Scene()

    # Swap buffers
    glutSwapBuffers()  


# Scene render function
def render_Scene():
    # Set current color to red
    glColor3f(1.0, 0.0, 0.0)

    for i in range(64):
        glPointSize(int(i/64*9+1))
        glColor3f(random.uniform(0,1), random.uniform(0,1), random.uniform(0,1))
        glBegin(GL_POINTS)
        glVertex2f(center[0] + radius * np.sin(2*np.pi*i/64), center[1] + radius * np.cos(2*np.pi*i/64))
        glEnd()


# Initialize GLUT
glutInit()

# Initialize the window with double buffering and RGB colors
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    
# Set the window size to 500x500 pixels
glutInitWindowSize(500, 500)

# Set the initial window position to (50, 50)
glutInitWindowPosition(glutGet(GLUT_SCREEN_HEIGHT)//2-250, glutGet(GLUT_SCREEN_WIDTH)//2-250)

# Create the window and give it a title
glutCreateWindow("Ali Khudiyev")

# Define callbacks
glutDisplayFunc(display)

# Begin event loop
glutMainLoop()
