
class Object:
    PLAYER = 0
    BALL = 1

    def __init__(self, x, y, type):
        self.x = x
        self.y = y
        self.speed = 1
        self.direction = (0, 0)
        self.type = type
    
    def move(self, direction, speed=5):
        self.x += speed * direction[0]
        self.y += speed * direction[1]
        self.direction = direction
        self.speed = speed
    
    def coords(self):
        return self.x, self.y

