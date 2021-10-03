import random
from object import *

class Ball(Object):
    def __init__(self, x, y, size=10):
        super().__init__(x, y, Object.BALL)
        self.speed = 5
        self.direction = random.choice([-1, 1]), random.choice([-1, 1])

        self.size = size
