from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


class Window:
    def __init__(self, width, height, title='Window', 
            display_callback=None, reshape_callback=None, keyboard_callback=None, special_callback=None,
            mouse_callback=None, motion_callback=None, passive_motion_callback=None, idle_callback=None):
        self.width = width
        self.height = height
        self.title = title

        self.callbacks = {
                'display': display_callback,
                'reshape': reshape_callback,
                'keyboard': keyboard_callback,
                'special': keyboard_callback,
                'mouse': mouse_callback,
                'motion': motion_callback,
                'passive_motion': passive_motion_callback,
                'idle': idle_callback,
                }

        glutInit()
        glutInitDisplayMode(GLUT_3_2_CORE_PROFILE, GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(50, 50)
        glutCreateWindow(self.title)

        glutDisplayFunc(self.callbacks['display'])
        glutReshapeFunc(self.callbacks['reshape'])
        glutKeyboardFunc(self.callbacks['keyboard'])
        glutSpecialFunc(self.callbacks['special'])
        glutMouseFunc(self.callbacks['mouse'])
        # glutMotionFunc(self.callbacks['motion'])
        # glutPassiveMotionFunc(self.callbacks['passive_motion'])
        glutIdleFunc(self.callbacks['idle'])

    def show(self):
        glutMainLoop()

