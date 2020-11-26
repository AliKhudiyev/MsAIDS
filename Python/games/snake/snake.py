from object import *


class Block:
    def __init__(self, x=0, y=0, old_dir=None, new_dir=None):
        self.x = x
        self.y = y
        self.size = 20
        self.old_dir = old_dir
        self.new_dir = new_dir
    
    def move(self, direction, speed):
        self.x += self.size * speed * direction[0]
        self.y += self.size * speed * direction[1]

        if self.new_dir is not None:
            self.old_dir = self.new_dir
        self.new_dir = direction

class Snake(Object):
    def __init__(self):
        super().__init__(Object.SNAKE)

        self.speed = 1
        self.blocks = [Block(50, 50)]
    
    def move(self, direction):
        self.blocks[0].move(direction, self.speed)
        for i, block in enumerate(self.blocks[1:]):
            block.move(self.blocks[i].old_dir, self.speed)
    
    def grow(self):
        # self.speed *= 1.1
        last_block = self.blocks[-1]
        block = Block(last_block.x - 20 * last_block.new_dir[0], last_block.y - 20 * last_block.new_dir[1])
        block.new_dir = last_block.old_dir
        self.blocks.append(block)
