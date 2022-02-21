# Importing the necessary Modules
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import numpy as np
import random

radius = 0.5
angle = np.pi/4


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

    # Draw a red square
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(0.0, 0.0)

    glVertex2f(radius*np.sin(0), radius*np.cos(0))
    glVertex2f(radius*np.sin(angle*1), radius*np.cos(angle*1))
    glVertex2f(radius*np.sin(angle*2), radius*np.cos(angle*2))
    glVertex2f(radius*np.sin(angle*3), radius*np.cos(angle*3))
    glVertex2f(radius*np.sin(angle*4), radius*np.cos(angle*4))
    glVertex2f(radius*np.sin(angle*5), radius*np.cos(angle*5))
    glVertex2f(radius*np.sin(angle*6), radius*np.cos(angle*6))
    glVertex2f(radius*np.sin(angle*7), radius*np.cos(angle*7))
    glVertex2f(radius*np.sin(angle*8), radius*np.cos(angle*8))
    glVertex2f(radius*np.sin(angle*9), radius*np.cos(angle*9))
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
