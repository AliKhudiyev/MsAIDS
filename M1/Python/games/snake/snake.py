from object import *


class Block:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.size = 20
        self.old_pos = x, y
        self.new_pos = x, y
    
    def move(self, position):
        self.x = position[0]
        self.y = position[1]

        if self.new_pos is not None:
            self.old_pos = self.new_pos
        self.new_pos = self.x, self.y


class Snake(Object):
    def __init__(self):
        super().__init__(Object.SNAKE)

        self.speed = 1
        self.blocks = [ Block(50, 50) ]
        self.counter = 1
    
    def move(self, direction):
        size = self.blocks[0].size
        for n in range(self.speed):
            self.blocks[0].move((self.blocks[0].x + size*direction[0], self.blocks[0].y + size*direction[1]))
            for i, block in enumerate(self.blocks[1:]):
                block.move(self.blocks[i].old_pos)
    
    def grow(self):
        if len(self.blocks) == 20 * self.counter:
            self.speed += 1
            self.counter += 2
        last_block = self.blocks[-1]
        block = Block(last_block.old_pos[0], last_block.old_pos[1])
        block.new_pos = last_block.old_pos
        self.blocks.append(block)
