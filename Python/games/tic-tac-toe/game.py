from tkinter import *
import numpy as np


class Game:
    X = 1
    O = 2
    Draw = 3

    def __init__(self):
        self.board = np.zeros(shape=(3,3))
        self.turn = Game.X

        self.root = Tk()
        self.root.title('Tic Tac Toe')

        self.canvas = Canvas(self.root, width=300, height=300)
        self.canvas.bind('<Button-1>', func=self.is_clicked)
        self.canvas.pack()

        self.frame = Frame(self.root)
        self.frame.pack()

        Button(self.frame, text='Quit', command=self.root.destroy).pack()

        self.canvas.create_line(100, 0, 100, 300)
        self.canvas.create_line(200, 0, 200, 300)
        self.canvas.create_line(0, 100, 300, 100)
        self.canvas.create_line(0, 200, 300, 200)

    def game_is_over(self):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] !=0:
                return self.board[i][0]
        
        for i in range(3):
            if self.board[0][i] == self.board[1][i] == self.board[2][i] !=0:
                return self.board[0][i]
        
        if self.board[0][0] == self.board[1][1] == self.board[2][2] !=0:
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] !=0:
            return self.board[0][2]
        
        if len(self.board[self.board != 0]) == 9:
            return Game.Draw

    def draw(self, x, y):
        if self.turn == Game.X:
            x0 = 100*x + 10
            y0 = 100*y + 10
            x1 = 100*(x+1) - 10
            y1 = 100*(y+1) - 10
            self.canvas.create_line(x0, y0, x1, y1)
            self.canvas.create_line(x0, y1, x1, y0)
        else:
            x0 = 100*x + 10
            y0 = 100*y + 10
            x1 = 100*(x+1) - 10
            y1 = 100*(y+1) - 10
            self.canvas.create_oval(x0, y0, x1, y1)

    def is_clicked(self, event):
        if self.turn == Game.Draw:
            return 0

        x, y = event.x // 100, event.y // 100
        if self.board[x][y] == 0:
            self.board[x][y] = self.turn
            self.draw(x, y)
            self.turn = self.turn % 2 + 1
        
        winner = self.game_is_over()
        if winner == Game.X:
            print('X won!')
            self.turn = Game.Draw
        elif winner == Game.O:
            print('O won!')
            self.turn = Game.Draw
        elif winner == Game.Draw:
            print('Draw!')
            self.turn = Game.Draw

    def update(self):
        self.root.after(100, func=self.update)

    def run(self):
        self.root.after(100, func=self.update)
        self.root.mainloop()


game = Game()
game.run()
