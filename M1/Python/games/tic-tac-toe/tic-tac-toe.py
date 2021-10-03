import numpy as np
import os


game_over = False
TIC = 1
TIC_sym = 'X'
TAC = 2
TAC_sym = 'O'
mod = TIC
board = np.zeros(shape=(3,3))


def is_over(board):
    for i in range(3):
        if (board[i][0] == board[i][1] == board[i][2] and board[i][0] != 0):
            return board[i][0]
        elif (board[0][i] == board[1][i] == board[2][i] and board[0][i] != 0):
            return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] != 0 or self.array[0][2] == self.array[1][1] == self.array[2][0] != 0:
        return board[1][1]
    
    if 0 in board:
        return 0
    return 3


def render(board):
    os.system('clear')
    print('  - - - - - - -')
    letters = ['A', 'B', 'C']
    for i in range(3):
        print(letters[i], end=' | ')
        for j in range(3):
            sym = ' '
            if board[i][j] == TIC:
                sym = TIC_sym
            elif board[i][j] == TAC:
                sym = TAC_sym
            print(sym, end=' | ')
        print('\n  - - - - - - -')
    print('    1   2   3')

while is_over(board) == 0:
    render(board)

    opponent = TIC_sym
    if mod == TAC:
        opponent = TAC_sym
    position = input(opponent+' turn: ')

    x, y = 0, 0
    if position.upper() == 'Q':
        break
    if len(position) == 2:
        x, y = ord(position[0].upper()) - ord('A'), ord(position[1]) - ord('1')
        if 0 <= x <= 2 and 0 <= y <= 2 and board[x][y] == 0:
            board[x][y] = mod
            if mod == TIC:
                mod = TAC
            elif mod == TAC:
                mod = TIC

render(board)
status = is_over(board)
if status == 3:
    print('Draw!')
elif status == TIC:
    print(TIC_sym+' won!')
elif status == TAC:
    print(TAC_sym+' won!')
