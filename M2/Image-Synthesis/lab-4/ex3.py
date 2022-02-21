from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

length = 1.0


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
    glEnd()

    # OY
    glColor3f(0.0, 1.0, 0.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, length, 0.0)
    glEnd()

    # OZ
    glColor3f(0.0, 0.0, 1.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, length)
    glEnd()

def idle():
    pass

def reshape(x, y):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(20.0, x/y, 4.5, 20)
    gluLookAt(2, -4, 2, 0, 0, 0, 0, 1, 0)
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
glutReshapeFunc(reshape)

# Begin event loop
glutMainLoop()
