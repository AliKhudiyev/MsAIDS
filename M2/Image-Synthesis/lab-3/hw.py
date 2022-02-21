# Ali Khudiyev

# Importing the necessary Modules
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import numpy as np
import time

move = (0.0, 0.0)
left, right, top, bottom = -0.3, 0.3, 0.3, -0.1


# Disply callback function
def display():
    # Reset background 
    glClear(GL_COLOR_BUFFER_BIT) 
    # Render scene 
    render_Scene()
    # Swap buffers
    glutSwapBuffers()

def idle():
    global move, left, right, top, bottom

    if left > 1.0:
        glTranslatef(-2.6, 0, 0)
        left -= 2.6
        right -= 2.6
    elif right < -1.0:
        glTranslatef(2.6, 0, 0)
        left += 2.6
        right += 2.6
    elif bottom > 1.0:
        glTranslatef(0, -2.4, 0)
        top -= 2.4
        bottom -= 2.4
    elif top < -1.0:
        glTranslatef(0, 2.4, 0)
        top += 2.4
        bottom += 2.4

    glTranslatef(move[0], move[1], 0)
    left += move[0]
    right += move[0]
    top += move[1]
    bottom += move[1]

    time.sleep(0.2)
    glutPostRedisplay()
    move = (0.0, 0.0)

# Scene render function
def render_Scene():
    glColor3f(1,0,0) 
    glBegin(GL_TRIANGLES) 
    glVertex2f(-0.3,-0.1) 
    glVertex2f(0.3,-0.1) 
    glVertex2f(0,0.3) 
    glEnd()

def keyboard_handler(char, x, y):
    print(f'Pressed {char} @ ({x}, {y})')

def special_keyboard_handler(char, x, y):
    global move

    if char == GLUT_KEY_UP:
        move = (0.0, 0.1)
    elif char == GLUT_KEY_DOWN:
        move = (0.0, -0.1)
    elif char == GLUT_KEY_RIGHT:
        move = (0.1, 0.0)
    elif char == GLUT_KEY_LEFT:
        move = (-0.1, 0.0)
    

# Initialize GLUT
glutInit()
# Initialize the window with double buffering and RGB colors 
glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
# Set the window size to 500x500 pixels 
glutInitWindowSize(500, 500)
# Set the initial window position to (50, 50) 
glutInitWindowPosition(50, 50)
# Create the window and give it a title
glutCreateWindow("Ali Khudiyev")
# Define display callback 
glutDisplayFunc(display)
# Define idle callback
glutIdleFunc(idle)
glutKeyboardFunc(keyboard_handler)
glutSpecialFunc(special_keyboard_handler)
# Begin event loop
glutMainLoop()
