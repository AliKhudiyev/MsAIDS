from window import *
from shader import *
from renderer import *
from renderable import *
from event import *
import time


window = None
shader = None
renderer = None

dx = 0

def display():
    global renderer
    global dx

    triangle = Triangle(-0.5+dx, -0.5, 0.0, 0.5, 0.5, -0.5, None)
    renderer.add(triangle)
    renderer.draw()

def idle():
    global dx
    # dx += 0.005
    time.sleep(1/60)
    glutPostRedisplay()

def mouse_handler(button, state, x, y):
    print('>', button, state, x, y)


# window initialization
window = Window(640, 640, 'LaYz', display_callback=display, mouse_callback=mouse_handler, 
        idle_callback=idle)

# glutMouseFunc(mouse_handler)
# glutMouseWheelFunc(mouse_wheel_handler)

# shader initialization
shader = Shader()
shader.load_vertex('vertex.glsl')
shader.load_fragment('fragment.glsl')
shader.compile()

# renderer initialization
renderer = Renderer(shader)

window.show()

