from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image

class Texture:
    def __init__(self, filepath):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)

    def enable(self):
        glBindTexture(GL_TEXTURE_2D, self.texture)

    def disable(self):
        glBindTexture(GL_TEXTURE_2D, 0)
