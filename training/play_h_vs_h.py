# play_h_vs_h.py

from src.board import print_board, check_triplets

def action_h(turn, positions):
    while True:
        if turn == 1:
            action = int(input('player 1, where to move? [1 to 9]:'))
        if turn == 2:
            action = int(input('player 2, where to move? [1 to 9]:'))
        if positions[action-1] == 0:
            return action
        else:
            print('spot already taken! please, choose a different spot on the board')
            continue
        
def play_h(turn):
    positions = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    while True:
        print_board(positions)
        if turn == 1:
            action = action_h(turn, positions)
            positions[action-1] = 1
            turn = 2
            if check_triplets(positions) == True:
                print_board(positions)
                print('player1 wins!')
                positions = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                break
            if check_triplets(positions) == 'Tie':
                print_board(positions)
                print('Tie!')
                positions = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                break
        if turn == 2:
            action = action_h(turn, positions)
            positions[action-1] = 2
            turn = 1
            if check_triplets(positions) == True:
                print_board(positions)
                print('player2 wins!')
                positions = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                break
            if check_triplets(positions) == 'Tie':
                print('Tie!')
                positions = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                break
            