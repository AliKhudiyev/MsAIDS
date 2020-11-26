import random
from object import *


class Fruit(Object):
    def __init__(self, width, height):
        super().__init__(Object.FRUIT)

        self.x = None
        self.y = None
        self.size = 16
        self.generate(width, height)
    
    def generate(self, width, height):
        self.x = random.randint(self.size/2, width-1-self.size/2)
        self.y = random.randint(self.size/2, height-1-self.size/2)
