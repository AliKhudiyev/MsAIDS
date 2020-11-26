from tkinter import *
from render import *
from snake import *
from fruit import *


class Game:
    def __init__(self):
        self.width = 500
        self.height = 500

        self.root = Tk()
        self.root.title('Snake 2D')
        self.root.bind('<Key>', func=self.key_pressed)

        self.canvas = Canvas(self.root, width=self.width, height=self.height, bg='black')
        self.canvas.pack()

        self.frame = Frame(self.root)
        self.frame.pack()

        self.button_start = Button(self.frame, text='Start/Stop', command=self.start)
        self.button_quit = Button(self.frame, text='Quit', command=self.root.destroy)

        self.button_start.pack()
        self.button_quit.pack()

        self.game_is_over = True
        self.direction = (0, 0)
        self.snake = Snake()
        self.fruit = Fruit(self.width, self.height)
        self.renderer = Renderer(self.canvas)

        self.renderer.submit([self.snake, self.fruit])

    def key_pressed(self, key):
        if key.char == 'a':
            self.direction = (-1, 0)
        elif key.char == 'd':
            self.direction = (1, 0)
        elif key.char == 'w':
            self.direction = (0, -1)
        elif key.char == 's':
            self.direction = (0, 1)

    def start(self):
        self.game_is_over = not self.game_is_over

    def handle_interaction(self):
        # Check int. between snake and walls

        # Check int. between snake and fruit
        if self.snake.blocks[0].x - 10 <= self.fruit.x <= self.snake.blocks[0].x + 10 and self.snake.blocks[0].y - 10 <= self.fruit.y <= self.snake.blocks[0].y + 10:
            # print('fruit:', self.fruit.x, self.fruit.y)
            self.snake.grow()
            self.fruit.generate(self.width, self.height)
        
        # Check int. between snake and snake
        pass

    def update(self):
        if not self.game_is_over:
            self.snake.move(self.direction)
            self.handle_interaction()
            self.renderer.flush()
        self.root.after(100, self.update)

    def run(self):
        self.root.after(100, self.update)
        self.root.mainloop()


game = Game()
game.run()
