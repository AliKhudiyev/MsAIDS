# Importing the necessary Modules
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Disply callback function
def display():
    # Reset background
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)

    # Render scene
    render_Scene()

    # Swap buffers
    glutSwapBuffers()  


# Scene render function
def render_Scene():
    # Set current color to red
    glColor3f(1.0,0.0,0.0)

    # Draw a red square
    glBegin(GL_POLYGON)
    glVertex2f(-0.5,-0.5)
    glVertex2f(-0.5,0.5)
    glVertex2f(0.5,0.5)
    glVertex2f(0.5,-0.5)
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
glutCreateWindow("My First OpenGL Window")

# Define callbacks
glutDisplayFunc(display)

# Begin event loop
glutMainLoop()
