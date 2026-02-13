# play_h_vs_h.py 

"""
This script allows two human players to play against each other.
- Players take turns inputting their moves.
- The game continues until one player wins or there is a tie.
"""

# import necessary libraries and modules
from src.board import print_board, check_triplets

def ask_int(prompt):
    while True:
        try:
            v = int(input(prompt))
            return v
        except ValueError:
            print("Per favore inserisci un numero intero.")

def action_h(turn, positions):
    while True:
        if turn == 1:
            action = ask_int('player 1, where to move? [1 to 9]: ')
        else:
            action = ask_int('player 2, where to move? [1 to 9]: ')

        if not (1 <= action <= 9):
            print('Valore fuori range. Inserisci un numero da 1 a 9.')
            continue

        idx = action - 1
        if positions[idx] == 0:
            return idx
        else:
            print('Spot già occupato! Scegli un’altra casella.')
            continue

def play_h(turn):
    positions = [0] * 9
    while True:
        print_board(positions)

        idx = action_h(turn, positions)
        positions[idx] = 1 if turn == 1 else 2
        turn = 2 if turn == 1 else 1

        outcome = check_triplets(positions)
        if outcome == True:
            print_board(positions)
            print('player1 wins!' if turn == 2 else 'player2 wins!')
            break
        if outcome == 'Tie':
            print_board(positions)
            print('Tie!')
            break