# play_to_train.py

"""
This script trains agents via self-play.
- Q-table agent learns via tabular Q-learning updates.  
"""

# import necessary libraries and modules
import numpy as np
import random
from training.play_random import Q1, memory
from src.board import check_triplets
from agents.Q_player import update_q_table, sel_e_greedy_action
from agents.DQN_player import select_action, update_model, hard_update_target
from tqdm import tqdm


# Q-LEARNING self-play

def action_train(Q, positions):
    a = sel_e_greedy_action(Q, positions, eps=0.1)
    return a

def play():
    num_episodes = 1_000_000
    positions = np.zeros(9, dtype=int)
    p1_win = p2_win = p_tie = p_tot = 0

    for _ in tqdm(range(num_episodes),
                  desc='Training...',
                  total=num_episodes, leave=True, ncols=80):

        p_tot += 1
        turn = random.randint(1, 2)

        while check_triplets(positions) == False:
            cur_pos = positions.copy()
            action = action_train(Q1, positions)
            if action is None:
                break
            positions[action] = 1 if turn == 1 else 2
            turn = 2 if turn == 1 else 1

        outcome = check_triplets(positions)
        if outcome == True and turn == 2:
            p1_win += 1
            update_q_table(Q1, cur_pos, action, 2, positions)
        elif outcome == 'Tie':
            p_tie += 1
            update_q_table(Q1, cur_pos, action, 1, positions)
        else:
            p2_win += 1
            update_q_table(Q1, cur_pos, action, -2, positions)

        positions[:] = 0

    # save Q-table as before
    with open('Q1.txt', 'w') as outfile:
        for key in sorted(Q1):
            outfile.write(str(key) + '\t' + str(Q1[key]) + '\n')

    print('------------------')
    print('Training finished!')
    print(f'player1 wins {((p1_win/p_tot)*100):.2f} % | '
          f'player2 wins {((p2_win/p_tot)*100):.2f} % | '
          f'tie {((p_tie/p_tot)*100):.2f} %')


# DQN self-play

def action_train_dqn(positions, epsilon):
    return select_action(positions, epsilon)

def play_dqn():
    num_episodes = 100_000
    positions = np.zeros(9, dtype=int)
    epsilon = 0.9

    p1_win = p2_win = p_tie = p_tot = 0

    for ep in tqdm(range(num_episodes),
                   desc='Training DQN...',
                   total=num_episodes, leave=True, ncols=80):

        p_tot += 1
        batch_size = min(p_tot, 64)
        turn = random.randint(1, 2)

        while check_triplets(positions) == False:
            s = positions.copy()

            action = action_train_dqn(positions, epsilon)
            if action is None:
                break

            step_reward = -1  # keep  original shaping

            positions[action] = 1 if turn == 1 else 2
            s_next = positions.copy()

            memory.append((s, action, step_reward, s_next, False))
            if len(memory) > batch_size:
                update_model(memory, batch_size)

            turn = 2 if turn == 1 else 1

        outcome = check_triplets(positions)
        if outcome == True and turn == 2:
            p1_win += 1
            terminal_reward = 2
        elif outcome == 'Tie':
            p_tie += 1
            terminal_reward = 2
        else:
            p2_win += 1
            terminal_reward = -2

        # terminal transition
        if len(memory) > 0:
            last_s, last_a, _, last_s_next, _ = memory[-1]
            memory.append((last_s, last_a, terminal_reward, positions.copy(), True))
            if len(memory) > batch_size:
                update_model(memory, batch_size)

        # epsilon decay with floor
        epsilon = max(0.05, epsilon * 0.95)

        # periodic target update
        if p_tot % 500 == 0:
            hard_update_target()

        positions[:] = 0

    print('------------------')
    print('Training finished!')
    print(f'player1 wins {((p1_win/p_tot)*100):.2f} % | '
          f'player2 wins {((p2_win/p_tot)*100):.2f} % | '
          f'tie {((p_tie/p_tot)*100):.2f} %')