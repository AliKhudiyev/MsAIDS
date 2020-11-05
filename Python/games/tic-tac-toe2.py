import numpy as np
import os


class Board:
    """
    Board class handles the board logic of the tic-tac-toe game.
    It also handles board-related actions such as rendering a board in console 
    or marking a position on the board and so on.
    """

    GAME_NOT_OVER = 0
    TIC = 1
    TIC_sym = 'X'
    TAC = 2
    TAC_sym = 'O'

    def __init__(self):
        self.array = np.zeros(shape=(3,3))
    
    # Get status of the board
    # Returns 0 if the game is not over
    # Returns 1 if TIC(X) is the winner
    # Returns 2 if TAC(O) is the winner
    # Returns 3 if there is no winner
    def status(self):
        for i in range(3):
            if (self.array[i][0] == self.array[i][1] == self.array[i][2] and self.array[i][0] != 0):
                # Winner is the one located at self.array[i][0]
                return self.array[i][0]
            elif (self.array[0][i] == self.array[1][i] == self.array[2][i] and self.array[0][i] != 0):
                # Winner is the one located at self.array[0][i]
                return self.array[0][i]
        if self.array[0][0] == self.array[1][1] == self.array[2][2] != 0 or self.array[0][2] == self.array[1][1] == self.array[2][0] != 0:
            # Winner is the one located at self.array[1][1]
            return self.array[1][1]
        
        if 0 in self.array:
            # The game is not over yet
            return Board.GAME_NOT_OVER
        
        # The game is draw
        return 3

    # Mark a position on the board
    # position - the 2D position given by "yx" (i.e. "a2")
    # mod - the player (i.e. TIC)
    def mark(self, position, mod):
        if len(position) != 2:
            return mod
        
        x, y = ord(position[0].upper()) - ord('A'), ord(position[1]) - ord('1')
        if 0 <= x <= 2 and 0 <= y <= 2 and self.array[x][y] == 0:
            self.array[x][y] = mod
            
            # switching the current mod
            if mod == Board.TIC:
                mod = Board.TAC
            elif mod == Board.TAC:
                mod = Board.TIC
        return mod
    
    # Renders the board in console
    def render(self):
        os.system('clear')
        print('  - - - - - - -')
        letters = ['A', 'B', 'C']
        for i in range(3):
            print(letters[i], end=' | ')
            for j in range(3):
                sym = ' '
                if self.array[i][j] == Board.TIC:
                    sym = Board.TIC_sym
                elif self.array[i][j] == Board.TAC:
                    sym = Board.TAC_sym
                print(sym, end=' | ')
            print('\n  - - - - - - -')
        print('    1   2   3')


class TicTacToe:
    """
    docstring
    """

    def __init__(self):
        self.board = Board()
    
    # Quits the game after rendering/printing the final result
    def quit(self):
        self.board.render()
        status = self.board.status()
        if status == 3:
            print('Draw!')
        elif status == Board.TIC:
            print(Board.TIC_sym+' won!')
        elif status == Board.TAC:
            print(Board.TAC_sym+' won!')
    
    # Launches the game
    def run(self):
        # TIC(X) begins first
        mod = Board.TIC

        # Game loop
        while self.board.status() == Board.GAME_NOT_OVER:
            self.board.render()

            opponent = Board.TIC_sym
            if mod == Board.TAC:
                opponent = Board.TAC_sym
            position = input(opponent+' turn: ')

            if position.upper() == 'Q':
                break
            mod = self.board.mark(position, mod)

        self.quit()


while True:
    option = input('Do you want to play Tic-Tac-Toe? [y/n] ')
    if option.upper() == 'Y':
        game = TicTacToe()
        game.run()
    else:
        break
