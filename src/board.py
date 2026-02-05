# board.py

import numpy as np
    
def print_board(positions):
    board = []
    for spot in positions:
        if spot == 0:
            board.append('_')
        if spot == 1:
            board.append('X')
        if spot == 2:
            board.append('O')
    
    board = np.asarray(board).reshape(3,3)

    print(board)

def check_triplets(positions):
    # check lines
    if positions[0]==positions[1] and positions[1]==positions[2] and positions[2] != 0:
        return True
    if positions[3]==positions[4] and positions[4]==positions[5] and positions[5] != 0:
        return True
    if positions[6]==positions[7] and positions[7]==positions[8] and positions[8] != 0:
        return True
    
    # check cols
    if positions[0]==positions[3] and positions[3]==positions[6] and positions[6] != 0:
        return True
    if positions[1]==positions[4] and positions[4]==positions[7] and positions[7] != 0:
        return True
    if positions[2]==positions[5] and positions[5]==positions[8] and positions[8] != 0:
        return True
    
    # check diag
    if positions[0]==positions[4] and positions[4]==positions[8] and positions[8] != 0:
        return True
    if positions[2]==positions[4] and positions[4]==positions[6] and positions[6] != 0:
        return True
    
    # check tie condition
    elif 0 not in positions:
        return 'Tie'
    else:
        return False
    





