# play_to_train.py

import numpy as np
import random
from training.play_random import Q1, memory
from src.board import check_triplets
from agents.Q_player import update_q_table, sel_e_greedy_action
from agents.DQN_player import select_action, update_model, hard_update_target
from tqdm import tqdm


def action_train(Q, positions):
    while True:
        action = sel_e_greedy_action(Q, positions)
        if positions[action] == 0:
            return action
        else:
            continue
        
def play():
    num_episodes = 1000000
    positions = np.zeros(9)
    p1_win, p2_win, p_tie, p_tot = 0, 0, 0, 0
    
    for _ in tqdm(range(num_episodes), 
                    desc='Training...', 
                    total=num_episodes,
                    leave=True,
                    ncols=80):
        p_tot +=1
        
        turn = random.randint(1,2)

        while check_triplets(positions) == False :
            cur_pos = positions.copy()
            if turn == 1:
                action = action_train(Q1, positions)
                positions[action] = 1
                turn = 2      
            else:
                action = action_train(Q1, positions)
                positions[action] = 2
                turn = 1
        
        if check_triplets(positions) == True and turn == 2:
            p1_win +=1
            update_q_table(Q1, cur_pos, action, 2, positions)
            # Reset in-place 
            positions[:] = 0
        elif check_triplets(positions) == 'Tie':
            p_tie +=1
            update_q_table(Q1, cur_pos, action, 1, positions)
            # Reset in-place 
            positions[:] = 0
        else:
            p2_win +=1
            update_q_table(Q1, cur_pos, action, -2, positions)
            # Reset in-place 
            positions[:] = 0
            
            
    outfile = open( 'Q1.txt', 'w' )
    for key in sorted(Q1):
        outfile.write( str(key) + '\t' + str(Q1[key]) + '\n' )
                
    print('------------------')              
    print('Training finished!')  
    print(f'player1 wins {((p1_win/p_tot)*100):.2f} % | player2 wins {((p2_win/p_tot)*100):.2f} % | tie {((p_tie/p_tot)*100):.2f} %')

    
def action_train_dqn(positions, memory, epsilon, cur_pos, batch_size):
    while True:
        action = select_action(positions, epsilon)
        if positions[action] == 0:
            return action
        else:
            memory.append((cur_pos, action, -3, positions, False))
            if len(memory) > batch_size:
                update_model(memory, batch_size)
            continue
    

        
def play_dqn():
    num_episodes = 100000
    positions = np.zeros(9)
    epsilon = 0.9
    p1_win, p2_win, p_tie, p_tot = 0, 0, 0, 0
    
    for _ in tqdm(range(num_episodes), 
                    desc='Training DQN...', 
                    total=num_episodes,
                    leave=True,
                    ncols=80):
        p_tot +=1

        batch_size = min(p_tot, 64)
        
        turn = random.randint(1,2)

        while check_triplets(positions) == False :
            cur_pos = positions.copy()
            if turn == 1:
                action = action_train_dqn(positions, memory, epsilon, cur_pos, batch_size)
                positions[action] = 1
                memory.append((cur_pos, action, -1, positions, False))
                turn = 2      
            else:
                action = action_train_dqn(positions, memory, epsilon, cur_pos, batch_size)
                positions[action] = 2
                memory.append((cur_pos, action, -1, positions, False))
                turn = 1
        
        if check_triplets(positions) == True and turn == 2:
            p1_win +=1
            memory.append((cur_pos, action, 2, positions, True))
            if len(memory) > batch_size:
                update_model(memory, batch_size)
                epsilon *= 0.95
            # Reset in-place 
            positions[:] = 0

        elif check_triplets(positions) == 'Tie':
            p_tie +=1
            memory.append((cur_pos, action, 2, positions, True))
            if len(memory) > batch_size:
                update_model(memory, batch_size)
                epsilon *= 0.95
            # Reset in-place 
            positions[:] = 0

        else:
            p2_win +=1
            memory.append((cur_pos, action, -2, positions, True))
            if len(memory) > batch_size:
                update_model(memory, batch_size)
                epsilon *= 0.95
            # Reset in-place 
            positions[:] = 0

        # Update target network every 500 episodes
        if p_tot % 500 == 0:
            hard_update_target()
                            
    print('------------------')              
    print('Training finished!')  
    print(f'player1 wins {((p1_win/p_tot)*100):.2f} % | player2 wins {((p2_win/p_tot)*100):.2f} % | tie {((p_tie/p_tot)*100):.2f} %')