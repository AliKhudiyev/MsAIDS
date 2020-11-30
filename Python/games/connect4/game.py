from tkinter import *
import numpy as np


class Game:
    Empty = 0
    Red = 1
    Blue = 2
    Draw = 3

    def __init__(self):
        self.n_row, self.n_col = 6, 7
        self.cell_size = 50

        self.root = Tk()
        self.root.title('Connect4')

        self.canvas = Canvas(self.root, width=self.n_col*self.cell_size+5, height=self.n_row*self.cell_size+5+50)
        self.canvas.bind('<Button-1>', func=self.is_clicked)
        self.canvas.bind('<Motion>', func=self.motion)
        self.canvas.pack()

        self.current_coin = None
        self.coins = []

        self.frame = Frame(self.root)
        self.frame.pack()
        
        Button(self.frame, text='New game', command=self.new_game).grid(row=1, column=0)
        Button(self.frame, text='Quit', command=self.root.destroy).grid(row=1, column=1)
        self.status = Label(self.frame)

        self.new_game()
    
    def new_game(self):
        self.board = np.zeros(shape=(self.n_row, self.n_col))
        self.turn = Game.Red
        self.canvas.delete('all')
        self.status.grid_forget()

        for i in range(self.n_row+1):
            x0 = 0
            y0 = self.cell_size*(i) + 50
            x1 = self.cell_size*self.n_col+3
            y1 = self.cell_size*(i) + 50
            self.canvas.create_line(x0, y0+3, x1, y1+3, width=1)
        
        for i in range(self.n_col+1):
            x0 = self.cell_size*(i)
            y0 = 50
            x1 = self.cell_size*(i)
            y1 = self.cell_size*self.n_row+3+50
            self.canvas.create_line(x0+3, y0, x1+3, y1)

    def game_is_over(self):
        for i in range(self.n_row-3):
            for j in range(self.n_col):
                if self.board[i][j] == self.board[i+1][j] == self.board[i+2][j] == self.board[i+3][j] != 0:
                    return self.board[i][j]
        
        for i in range(self.n_row):
            for j in range(self.n_col-3):
                if self.board[i][j] == self.board[i][j+1] == self.board[i][j+2] == self.board[i][j+3] != 0:
                    return self.board[i][j]
        
        for i in range(3, self.n_row):
            for j in range(self.n_col-3):
                if self.board[i][j] == self.board[i-1][j+1] == self.board[i-2][j+2] == self.board[i-3][j+3] != 0:
                    return self.board[i][j]
        
        for i in range(self.n_row-3):
            for j in range(self.n_col-3):
                if self.board[i][j] == self.board[i+1][j+1] == self.board[i+2][j+2] == self.board[i+3][j+3] != 0:
                    return self.board[i][j]
        
        if len(self.board[self.board != 0]) == len(self.board.flatten(order='C')):
            return Game.Draw

        return Game.Empty

    def is_clicked(self, event):
        if self.turn == Game.Draw:
            return 0
        
        x = event.x // self.cell_size
        invalid_place = True
        for i in range(self.n_row):
            if self.board[self.n_row-1-i][x] == Game.Empty:
                self.board[self.n_row-1-i][x] = self.turn
                invalid_place = False
                break
        
        if not invalid_place:
            self.turn = self.turn % 2 + 1
            self.motion(event)
        
        winner = self.game_is_over()
        if winner == Game.Red:
            self.turn = Game.Draw
            self.status.config(text='Red won!')
            self.status.grid(row=0, column=0, columnspan=2)
        elif winner == Game.Blue:
            self.status.config(text='Blue won!')
            self.status.grid(row=0, column=0, columnspan=2)
            self.turn = Game.Draw
        elif winner == Game.Draw:
            self.status.config(text='Draw!')
            self.status.grid(row=0, column=0, columnspan=2)
            self.turn = Game.Draw
    
    def motion(self, event):
        # print(event.x, event.y)
        if self.current_coin is not None:
            self.canvas.delete(self.current_coin)
        color = 'red'
        if self.turn == Game.Blue:
            color = 'blue'
        x = event.x // self.cell_size
        self.current_coin = self.canvas.create_oval(self.cell_size*x+5+3, 5, self.cell_size*x+45+3, 45, fill=color, outline=color)
    
    def draw(self):
        if len(self.coins):
            for coin in self.coins:
                self.canvas.delete(coin)
        
        self.coins = []
        for i in range(self.n_row):
            for j in range(self.n_col):
                if self.board[i][j] == Game.Empty:
                    continue

                color = 'blue'
                if self.board[i][j] == Game.Red:
                    color = 'red'
                
                x0 = self.cell_size*j
                y0 = self.cell_size*i
                self.coins.append(self.canvas.create_oval(x0+3+10, y0+50+3+10, x0+50+3-10, y0+100+3-10, fill=color, outline=color))

    def update(self):
        self.draw()
        self.root.after(100, func=self.update)

    def run(self):
        self.root.after(100, func=self.update)
        self.root.mainloop()


game = Game()
game.run()