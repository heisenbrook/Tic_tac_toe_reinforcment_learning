# play_h_vs_bot.py

"""
Define a human vs bot game mode for Tic Tac Toe, allowing a human player to compete against
a trained Q-learning or DQN agent.
The human player can choose to play against either a Q-table based agent or a DQN-based agent,
with the game providing feedback on the board state and outcomes after each move.
"""

# import necessary libraries
import numpy as np
from src.board import print_board, check_triplets
from agents.Q_player import sel_e_greedy_action
from training.play_random import Q1
from agents.DQN_player import select_action

def act_q(Q, turn, positions):
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
        
def play_h_vs_q(turn):
    positions = np.zeros(9)
    Q = Q1

    while check_triplets(positions) == False :
        if turn == 1:
            action = act_q(Q, turn, positions)
            positions[action] = 1
            turn = 2      
        else:
            action = act_q(Q, turn, positions)
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
        
def act_dqn(turn, positions):
    while True:
        if turn == 1:
            action = int(input('player 1, where to move? [0 to 8]:'))
        if turn == 2:
            action = select_action(positions, 0)
        if positions[action] == 0:
            return action
        else:
            print('spot already taken! please, choose a different spot on the board')
            continue
        
def play_vs_dqn(turn):
    positions = np.zeros(9)

    while check_triplets(positions) == False :
        if turn == 1:
            action = act_dqn(turn, positions)
            positions[action] = 1
            turn = 2      
        else:
            action = act_dqn(turn, positions)
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
            