import numpy as np
from ball import *
from player import *
from render import *


class Pong:
    def __init__(self):
        self.width, self.height = 600, 400

        self.root = Tk()
        self.root.title('Pong')

        self.canvas = Canvas(self.root, width=self.width, height=self.height, bg='black')
        self.canvas.pack()

        self.frame = Frame(self.root)
        self.frame.pack()

        self.button_start = Button(self.root, text='Start/Stop', command=self.start)
        self.button_quit = Button(self.root, text='Quit', command=self.root.destroy)

        self.button_start.pack()
        self.button_quit.pack()

        self.game_is_over = True

        self.player1_score = 0
        self.player2_score = 0

        self.renderer = Renderer(self.canvas)
        self.player1 = Player(20, 200)
        self.player2 = Player(580, 200)
        self.ball = Ball(300, 200)

        self.root.bind('<Key>', lambda key: self.key_pressed(key))
        self.root.bind('<Up>', lambda key: self.key_pressed(key, arrow='w'))
        self.root.bind('<Down>', lambda key: self.key_pressed(key, arrow='s'))
        self.renderer.submit([self.player1, self.player2, self.ball])

    def key_pressed(self, key, arrow=''):
        if key.char == 'q':
            self.player1.move(direction=(0,-1), speed=12)
            self.player1.acceleration = -0.25
        elif key.char == 'a':
            self.player1.move(direction=(0,1), speed=12)
            self.player1.acceleration = 0.25
        else:
            self.player2.acceleration = 0

        if arrow == 'w':
            self.player2.move(direction=(0,-1), speed=12)
            self.player2.acceleration = -0.25
        elif arrow == 's':
            self.player2.move(direction=(0,1), speed=12)
            self.player2.acceleration = 0.25
        else:
            self.player1.acceleration = 0

    def start(self):
        self.game_is_over = not self.game_is_over

    def handle_interactions(self):
        ball_coords = self.ball.coords()
        ball_direction = self.ball.direction
        direction = ball_direction

        player1_coords = self.player1.coords()
        player2_coords = self.player2.coords()
        
        if ball_coords[0] <= player1_coords[0]:
            if player1_coords[1] - 25 <= ball_coords[1] <= player1_coords[1] + 25:
                direction = -1 * ball_direction[0], ball_direction[1] + self.player1.acceleration
                self.ball.speed += 0.1 * self.ball.speed
        elif ball_coords[0] >= player2_coords[0]:
            if player2_coords[1] - 25 <= ball_coords[1] <= player2_coords[1] + 25:
                direction = -1 * ball_direction[0], ball_direction[1] + self.player2.acceleration
                self.ball.speed += 0.1 * self.ball.speed
        
        if ball_coords[1] - self.ball.size / 2 <= 0 or ball_coords[1] + self.ball.size / 2 >= self.height:
            direction = self.ball.direction[0], -1 * self.ball.direction[1]

        self.ball.move(direction, speed=self.ball.speed)

        if player1_coords[1] - 25 <= 0:
            self.player1.y = 25
        elif player1_coords[1] + 25 >= self.height:
            self.player1.y = self.height-25
        if player2_coords[1] - 25 <= 0:
            self.player2.y = 25
        elif player2_coords[1] + 25 >= self.height:
            self.player2.y = self.height-25

    def update(self):
        if not self.game_is_over:
            ball_coords = self.ball.coords()
            if self.ball.x + 5 >= self.width:
                self.player1_score += 1
                self.ball = Ball(self.width/2, self.height/2)
                self.renderer.submit([self.ball, self.player1, self.player2])
            elif self.ball.x - 5 <= 0:
                self.player2_score += 1
                self.ball = Ball(self.width/2, self.height/2)
                self.renderer.submit([self.ball, self.player1, self.player2])
            
            self.handle_interactions()
            self.renderer.flush()

            self.canvas.create_text(self.width/4, 15, text='Player 1: ' + str(self.player1_score), fill='white')
            self.canvas.create_text(3*self.width/4, 15, text='Player2: ' + str(self.player2_score), fill='white')
        self.root.after(100, func=self.update)
    
    def run(self):
        self.root.after(100, func=self.update)
        self.root.mainloop()



game = Pong()
game.run()

# root = Tk()
# root.title('Pong')

# canvas = Canvas(root, width=600, height=400, bg='black')
# canvas.pack()

# frame = Frame(root)
# frame.pack()

# game_is_over = True
# ball = None
# dx_sign, dy_sign = 2*random.randint(0, 1)-1, 2*random.randint(0, 1)-1
# dx = random.randint(1, 10)
# dy = random.randint(1, 10)


# def update():
#     global dx_sign, dy_sign
#     global dx, dy

#     if not game_is_over:
#         coords = canvas.coords(ball)

#         if coords[2] >= 600:
#             dx_sign = -1
#         elif coords[0] <= 0:
#             dx_sign = 1
#         if coords[3] >= 400:
#             dy_sign = -1
#         elif coords[1] <= 0:
#             dy_sign = 1
        
#         canvas.move(ball, dx_sign*dx, dy_sign*dy)

#     root.after(100, func=update)

# def start():
#     global game_is_over
#     game_is_over = not game_is_over

# def quit():
#     root.destroy()


# ball = canvas.create_oval(75, 75, 100, 100, fill='red', outline='red')

# button_start = Button(frame, text='Start/Stop', command=start)
# button_quit = Button(frame, text='Quit', command=quit)
# button_start.pack()
# button_quit.pack()

# root.after(100, func=update)
# root.mainloop()
