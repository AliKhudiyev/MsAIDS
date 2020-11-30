import numpy as np
import os


class Board:
    """
    Board class handles the things related to the board.
    (i.e. rendering in console, current status)

    It also handles the board logic in the connect4 game.
    """

    GAME_NOT_OVER = 0
    RED = 1
    BLUE = 2
    RED_sym = 'R'
    BLUE_sym = 'B'
    VERTICAL_BEGIN = 'A'

    # 4 <= width, height <= 10
    def __init__(self, width, height):
        if width > 10:
            width = 10
        elif width < 4:
            width = 4
        if height > 10:
            height = 10
        if height < 4:
            height = 4

        self.width = width
        self.height = height
        self.array = np.zeros(shape=(height, width))

    # Give status of the board
    # Returns 0 if the game is not over
    # Returns 1 if the Red is winner
    # Returns 2 if the Blue is winner
    # Returns 3 if there is no winner
    def status(self):
        for i in range(self.height):
            for j in range(self.width-4):
                if self.array[i][j] == self.array[i][j+1] == self.array[i][j+2] == self.array[i][j+3] != 0:
                    # Winner is the one located at self.array[i][j]
                    return self.array[i][j]
        
        for i in range(self.height-4):
            for j in range(self.width):
                if self.array[i][j] == self.array[i+1][j] == self.array[i+2][j] == self.array[i+3][j] != 0:
                    # Winner is the one located at self.array[i][j]
                    return self.array[i][j]
        
        for i in range(3, self.height):
            for j in range(self.width-3):
                if self.array[i][j] == self.array[i-1][j+1] == self.array[i-2][j+2] == self.array[i-3][j+3] != 0:
                    # Winner is the one located at self.array[i][j]
                    return self.array[i][j]
        
        for i in range(self.height-3):
            for j in range(self.width-3):
                if self.array[i][j] == self.array[i+1][j+1] == self.array[i+2][j+2] == self.array[i+3][j+3] != 0:
                    # Winner is the one located at self.array[i][j]
                    return self.array[i][j]
        
        if 0 in self.array:
            # The game is not over yet
            return Board.GAME_NOT_OVER

        # The game is draw
        return 3

    # Put a Red/Blue coin in the grid
    # position - horizontal position given in the x axis (i.e. 3)
    # mod - the color of coin (i.e. Board.Red)
    def mark(self, position, mod):
        try:
            x, y = int(position)-1, self.height-1
        except ValueError:
            return mod
        
        if len(position) != 1 or x < 0 or x >= self.width:
            return mod
        
        for i in range(self.height):
            if self.array[y][x] != 0:
                y -= 1
        
        if 0 <= y < self.height:
            self.array[y][x] = mod

            # switching the current mod
            if mod == Board.RED:
                mod = Board.BLUE
            elif mod == Board.BLUE:
                mod = Board.RED
        
        return mod

    # Renders the board in console
    def render(self):
        os.system('clear')
        print('  ', end=' ')
        for i in range(self.width):
            print('- -', end=' ')
        print()
        for i in range(self.height):
            print(chr(ord(Board.VERTICAL_BEGIN)+i), end=' | ')
            for j in range(self.width):
                sym = ' '
                if self.array[i][j] == Board.RED:
                    sym = '\033[1;31;40m' + Board.RED_sym + '\033[0m'
                elif self.array[i][j] == Board.BLUE:
                    sym = '\033[1;34;40m' + Board.BLUE_sym + '\033[0m'
                print(sym, end=' | ')
            print('\n  ', end=' ')
            for i in range(self.width):
                print('- -', end=' ')
            print()
        print(' ', end=' ')
        for i in range(self.width):
            print('  '+str(i+1), end=' ')
        print()


class Connect4:
    """
    This class is to handle game initialization and user inputs, 
    as well as the game logic.
    """

    def __init__(self, width, height):
        self.board = Board(width, height)

    # Quits the game after rendering/printing the final result
    def quit(self):
        self.board.render()
        status = self.board.status()
        if status == 3:
            print('Draw!')
        elif status == Board.RED:
            print('RED won!')
        elif status == Board.BLUE:
            print('BLUE won!')

    # Launches the game
    def run(self):
        # Red begins first
        mod = Board.RED

        # Game loop
        while self.board.status() == Board.GAME_NOT_OVER:
            self.board.render()

            sym = Board.RED_sym
            if mod == Board.BLUE:
                sym = Board.BLUE_sym
            position = input(sym+' turn: ')

            if position.upper() == 'Q':
                break
            mod = self.board.mark(position, mod)
        
        self.quit()


while True:
    option = input('Do you want to play connect4? [y/n] ')
    if option.upper() == 'Y':
        game = Connect4(7, 6)
        game.run()
    else:
        break
