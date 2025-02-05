import numpy as np
from utils.board import print_board, check_triplets
from utils.Q_player import sel_e_greedy_action
from utils.play_random import Q1

def act(Q, turn, positions):
    while True:
        if turn == 1:
            action = int(input('player 1, where to move? [0 to 8]:'))
        if turn == 2:
            action = sel_e_greedy_action(Q, positions)
        if positions[action] == 0:
            return action
        else:
            print('spot already taken! please, choose a different spot on the board')
            continue
        
def play_h_vs_b(turn):
    positions = np.zeros(9)
    Q = Q1

    while check_triplets(positions) == False :
        if turn == 1:
            action = act(Q, turn, positions)
            positions[action] = 1
            turn = 2      
        else:
            action = act(Q, turn, positions)
            positions[action] = 2
            turn = 1
            
            print_board(positions)
        
    if check_triplets(positions) == True and turn == 2:
        print('player1 wins!')
        print_board(positions)
        positions = np.zeros(9)
    elif check_triplets(positions) == 'Tie':
        print('Tie!')
        print_board(positions)
        positions = np.zeros(9)
    else:
        print('player2 wins!')
        print_board(positions)
        positions = np.zeros(9)
            