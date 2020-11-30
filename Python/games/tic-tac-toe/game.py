from tkinter import *
import numpy as np


class Game:
    X = 1
    O = 2
    Draw = 3

    def __init__(self):
        self.root = Tk()
        self.root.title('Tic Tac Toe')

        self.canvas = None
        self.board = None
        self.turn = None
        self.frame = None
        self.status = None

        self.new_game()

    def new_game(self):
        if self.canvas is not None:
            self.canvas.pack_forget()
        if self.frame is not None:
            self.frame.pack_forget()
        
        self.board = np.zeros(shape=(3,3))
        self.turn = Game.X

        self.canvas = Canvas(self.root, width=300, height=300)
        self.canvas.bind('<Button-1>', func=self.is_clicked)
        self.canvas.pack()

        self.frame = Frame(self.root)
        self.frame.pack()

        self.status = Label(self.frame)
        Button(self.frame, text='New game', command=self.new_game).grid(row=1, column=0)
        Button(self.frame, text='Quit', command=self.root.destroy).grid(row=1, column=1)

        self.canvas.create_line(100, 0, 100, 300)
        self.canvas.create_line(200, 0, 200, 300)
        self.canvas.create_line(0, 100, 300, 100)
        self.canvas.create_line(0, 200, 300, 200)

    def game_is_over(self):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] !=0:
                self.turn = Game.Draw
                self.draw(i, 0, 'red', self.board[i][0])
                self.draw(i, 1, 'red', self.board[i][0])
                self.draw(i, 2, 'red', self.board[i][0])
                return self.board[i][0]
        
        for i in range(3):
            if self.board[0][i] == self.board[1][i] == self.board[2][i] !=0:
                self.turn = Game.Draw
                self.draw(0, i, 'red', self.board[0][i])
                self.draw(1, i, 'red', self.board[0][i])
                self.draw(2, i, 'red', self.board[0][i])
                return self.board[0][i]
        
        if self.board[0][0] == self.board[1][1] == self.board[2][2] !=0:
            self.turn = Game.Draw
            self.draw(0, 0, 'red', self.board[1][1])
            self.draw(1, 1, 'red', self.board[1][1])
            self.draw(2, 2, 'red', self.board[1][1])
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] !=0:
            self.turn = Game.Draw
            self.draw(0, 2, 'red', self.board[1][1])
            self.draw(1, 1, 'red', self.board[1][1])
            self.draw(2, 0, 'red', self.board[1][1])
            return self.board[0][2]
        
        if len(self.board[self.board != 0]) == 9:
            self.turn = Game.Draw
            return Game.Draw

    def draw(self, x, y, color='black', turn=None):
        if self.turn == Game.X or turn == Game.X:
            x0 = 100*x + 10
            y0 = 100*y + 10
            x1 = 100*(x+1) - 10
            y1 = 100*(y+1) - 10
            self.canvas.create_line(x0, y0, x1, y1, fill=color, width=5)
            self.canvas.create_line(x0, y1, x1, y0, fill=color, width=5)
        elif self.turn == Game.O or turn == Game.O:
            x0 = 100*x + 10
            y0 = 100*y + 10
            x1 = 100*(x+1) - 10
            y1 = 100*(y+1) - 10
            self.canvas.create_oval(x0, y0, x1, y1, outline=color, width=5)

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
            Label(self.frame, text='Player 1 (X) won!', fg='red').grid(row=0, column=0, columnspan=2)
        elif winner == Game.O:
            Label(self.frame, text='Player 2 (O) won!', fg='red').grid(row=0, column=0, columnspan=2)
        elif winner == Game.Draw:
            Label(self.frame, text='Draw!', fg='red').grid(row=0, column=0, columnspan=2)

    def update(self):
        if self.turn != Game.Draw:
            self.status.config(text='Player '+str(self.turn)+' is playing...')
            self.status.grid(row=0, column=0, columnspan=2)
        else:
            self.status.grid_forget()
        self.root.after(100, func=self.update)

    def run(self):
        self.root.after(100, func=self.update)
        self.root.mainloop()


game = Game()
game.run()
