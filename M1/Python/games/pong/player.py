from object import *


class Player(Object):
    def __init__(self, x, y, width=5, height=50):
        super().__init__(x, y, Object.PLAYER)
        self.speed = 8
        self.direction = (0, 0)
        self.acceleration = 0

        self.width = width
        self.height = height

