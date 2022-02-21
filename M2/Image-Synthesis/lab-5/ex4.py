# Program that draws a solid Cone with lights
# Adapted by Ammar Assoum

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import time

# rotation axis
axis = [0.0, 0.0, 0.0]

# rotation angle
angle = -90.0

# rotation speed
speed = 1

# light1 position
light_pos = 0.5

def background():
    # Set the background color of the window to Gray
    glClearColor(0.5, 0.5, 0.5, 0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

def perspective():
    # establish the projection matrix (perspective)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # Get the viewport  to use it in chosing the aspect ratio of gluPerspective
    _,_,width,height = glGetDoublev(GL_VIEWPORT) # we don't need x and y
    gluPerspective(45,width/height,0.25,200)

def lookat():
    # and then the model view matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(0,-1,6,0,0,0,0,1,0)

def light():
    #Setup light 0 and enable lighting
    glLightfv(GL_LIGHT0, GL_AMBIENT, GLfloat_4(0.0, 1.0, 0.0, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, GLfloat_4(1.0, 1.0, 1.0, 1.0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, GLfloat_4(1.0, 1.0, 1.0, 1.0))
    glLightfv(GL_LIGHT0, GL_POSITION, GLfloat_4(1.0, 1.0, 1.0, 0.0))
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, GLfloat_4(0.2, 0.2, 0.2, 1.0))
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

	#Setup light 1 and enable lighting
    glLightfv(GL_LIGHT1, GL_AMBIENT, GLfloat_4(light_pos, light_pos, light_pos, 1.0))
    glLightfv(GL_LIGHT1, GL_DIFFUSE, GLfloat_4(light_pos, light_pos, light_pos, 1.0))
    glLightfv(GL_LIGHT1, GL_SPECULAR, GLfloat_4(light_pos, light_pos, light_pos, 1.0))
    glLightfv(GL_LIGHT1, GL_POSITION, GLfloat_4(light_pos, light_pos, light_pos, 1.0));   
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, GLfloat_4(0.2, 0.2, 0.2, 1.0))
    glEnable(GL_LIGHT1)

def depth():
	#Setup depth testing
	glEnable(GL_DEPTH_TEST)
	glDepthFunc(GL_LESS)    

def coneMaterial():
	#Setup material for cone
	glMaterialfv(GL_FRONT, GL_AMBIENT, GLfloat_4(0.2, 0.2, 0.2, 1.0))
	glMaterialfv(GL_FRONT, GL_DIFFUSE, GLfloat_4(0.8, 0.8, 0.8, 1.0))
	glMaterialfv(GL_FRONT, GL_SPECULAR, GLfloat_4(1.0, 0.0, 1.0, 1.0))
	glMaterialfv(GL_FRONT, GL_SHININESS, GLfloat(50.0))

def transformations():
    global angle, axis, speed

    glRotated(angle, axis[0], axis[1], axis[2])
    angle += speed * 5.0

def drawCone(radius, height, slices,stacks):
    glPushMatrix()
    glutSolidCone(radius, height, slices, stacks )
    glPopMatrix()   
    
def display():
    background()
    perspective()
    lookat()
    light()
    depth()
    coneMaterial()
    transformations()
    drawCone(1,2,50,10)
    glutSwapBuffers()

def idle():
    time.sleep(0.2)
    glutPostRedisplay()

def keyboard_handler(key, x, y):
    global axis, angle, speed

    if key == 'x'.encode('ascii'):
        axis[0] += 0.1
    elif key == 'y'.encode('ascii'):
        axis[1] += 0.1
    elif key == 'z'.encode('ascii'):
        axis[2] += 0.1
    elif key == 'q'.encode('ascii'):
        speed += 0.1
    elif key == 'a'.encode('ascii'):
        speed -= 0.1

def special_key_handler(key, x, y):
    global light_pos

    if key == GLUT_KEY_F1:
        light_pos += 0.1
    elif key == GLUT_KEY_F2:
        light_pos -= 0.1

# Initialize GLUT
glutInit()

# Initialize the window with double buffering and RGB colors
glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    
# Set the window size to 500x500 pixels
glutInitWindowSize(500, 500)

# Create the window and give it a title
glutCreateWindow("Ali Khudiyev")

glClearColor(0.0,0.0,0.0,0.0)

# Set the initial window position to (50, 50)
glutInitWindowPosition(50, 50)

# Define display callback
glutDisplayFunc(display)

# Defining idle function
glutIdleFunc(idle)

# Registering keyboard handlers
glutKeyboardFunc(keyboard_handler)
glutSpecialFunc(special_key_handler)
 
# Begin event loop
glutMainLoop()
