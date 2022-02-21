# Ali Khudiyev

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import time, random

length = 1.0
arrow_dim = None
axe = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))


def display():
    glClear(GL_COLOR_BUFFER_BIT)
    render_scene()
    glutSwapBuffers()

def render_scene():
    global length

    # OX
    glColor3f(1.0, 0.0, 0.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(length, 0.0, 0.0)

    glVertex3f(0.75, 0.25, 0.0)
    glVertex3f(length, 0.0, 0.0)
    glVertex3f(length, 0.0, 0.0)
    glVertex3f(0.75, -0.25, 0.0)
    glVertex3f(0.75, -0.25, 0.0)
    glVertex3f(0.75, 0.25, 0.0)

    glVertex3f(0.75, 0.0, 0.25)
    glVertex3f(length, 0.0, 0.0)
    glVertex3f(length, 0.0, 0.0)
    glVertex3f(0.75, 0.0, -0.25)
    glVertex3f(0.75, 0.0, -0.25)
    glVertex3f(0.75, 0.0, 0.25)
    glEnd()

    # OY
    glColor3f(0.0, 1.0, 0.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, length, 0.0)

    glVertex3f(0.25, 0.75, 0.0)
    glVertex3f(0.0, length, 0.0)
    glVertex3f(0.0, length, 0.0)
    glVertex3f(-0.25, 0.75, 0.0)
    glVertex3f(-0.25, 0.75, 0.0)
    glVertex3f(0.25, 0.75, 0.0)

    glVertex3f(0.0, 0.75, 0.25)
    glVertex3f(0.0, length, 0.0)
    glVertex3f(0.0, length, 0.0)
    glVertex3f(0.0, 0.75, -0.25)
    glVertex3f(0.0, 0.75, -0.25)
    glVertex3f(0.0, 0.75, 0.25)
    glEnd()

    # OZ
    glColor3f(0.0, 0.0, 1.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, length)

    glVertex3f(0.0, 0.25, 0.75)
    glVertex3f(0.0, 0.0, length)
    glVertex3f(0.0, 0.0, length)
    glVertex3f(0.0, -0.25, 0.75)
    glVertex3f(0.0, -0.25, 0.75)
    glVertex3f(0.0, 0.25, 0.75)

    glVertex3f(0.25, 0.0, 0.75)
    glVertex3f(0.0, 0.0, length)
    glVertex3f(0.0, 0.0, length)
    glVertex3f(-0.25, 0.0, 0.75)
    glVertex3f(-0.25, 0.0, 0.75)
    glVertex3f(0.25, 0.0, 0.75)
    glEnd()

def idle():
    glRotated(10, axe[0], axe[1], axe[2])
    # glTranslated(random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3))
    time.sleep(0.1)
    glutPostRedisplay()

def reshape(x, y):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(50.0, x/y, 0.1, 30)
    gluLookAt(3, 3, 3, 0, 0, 0, 0, 1, 0)
    glViewport(0, 0, x, y)


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
# Define window reshape callback
glutReshapeFunc(reshape)
# Begin event loop
glutMainLoop()

'''
Q: What will happen if we combine the rotation process with translation without any control on it? Propose a solution.

A: Without any control, translation could cause the axis get clipped out at at some frames. To avoid it, we could develop physics to be able to detect any anomalies or cases that have to be handled immediately.
'''
