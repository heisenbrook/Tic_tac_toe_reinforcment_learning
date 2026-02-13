# play_random.py

"""
This script trains a Q-table agent and a DQN agent against a random opponent.
- Q-table agent learns via tabular Q-learning updates.
- DQN agent learns via experience replay and target network updates.
The random opponent selects uniformly from available moves.
"""


# import necessary libraries and modules
import random
from tqdm import tqdm
import numpy as np
from collections import deque
from src.board import check_triplets
from agents.Q_player import update_q_table, sel_e_greedy_action
from agents.DQN_player import select_action, update_model, hard_update_target

# Q-table and replay memory
Q1 = {}
memory = deque(maxlen=10_000_000)


# Helpers

def random_action_from_empty(positions):
    empties = np.where(positions == 0)[0]
    if len(empties) == 0:
        return None
    return int(np.random.choice(empties))


# Q-LEARNING vs RANDOM

def action_r(Q, turn, positions):
    """
    turn 1: Q-table agent
    turn 2: random agent
    Return an int action for an empty cell.
    """
    if turn == 1:
        a = sel_e_greedy_action(Q, positions, eps=0.1)
    else:
        a = random_action_from_empty(positions)
    return a

def play_r():
    num_episodes = 1_000_000
    positions = np.zeros(9, dtype=int)
    p1_win = p2_win = p_tie = p_tot = 0

    for _ in tqdm(range(num_episodes),
                  desc='Training with random Q1',
                  total=num_episodes, leave=True, ncols=80):

        p_tot += 1
        turn = random.randint(1, 2)
        # episode loop
        while check_triplets(positions) == False:
            cur_pos = positions.copy()
            action = action_r(Q1, turn, positions)
            if action is None:
                break  # board full (should be handled by tie check)

            # apply action
            positions[action] = 1 if turn == 1 else 2
            # switch turn
            turn = 2 if turn == 1 else 1

        # terminal evaluation
        outcome = check_triplets(positions)
        if outcome == True and turn == 2:  # last move by player1
            p1_win += 1
            # LAST (s,a) that led to terminal isn't tracked here step-by-step;
            # keep your original "update last seen" semantics
            update_q_table(Q1, cur_pos, action, 2, positions)
        elif outcome == 'Tie':
            p_tie += 1
            update_q_table(Q1, cur_pos, action, 1, positions)
        else:
            p2_win += 1
            update_q_table(Q1, cur_pos, action, -2, positions)

        # reset in-place
        positions[:] = 0

    print('------------------')
    print('Training random finished!')
    print(f'player1 wins {((p1_win/p_tot)*100):.2f} | '
          f'player2 wins {((p2_win/p_tot)*100):.2f} | '
          f'tie {((p_tie/p_tot)*100):.2f}')


# DQN vs RANDOM

def action_r_dqn(turn, positions, epsilon):
    """
    turn 1: DQN agent (epsilon-greedy, masked)
    turn 2: random agent
    Returns int action.
    """
    if turn == 1:
        a = select_action(positions, epsilon)
    else:
        a = random_action_from_empty(positions)
    return a

def play_r_dqn():
    num_episodes = 100_000
    positions = np.zeros(9, dtype=int)

    epsilon = 0.9
    p1_win = p2_win = p_tie = p_tot = 0

    for ep in tqdm(range(num_episodes),
                   desc='Training with random DQN',
                   total=num_episodes, leave=True, ncols=80):

        p_tot += 1
        batch_size = min(p_tot, 64)
        turn = random.randint(1, 2)

        # episode loop
        while check_triplets(positions) == False:
            s = positions.copy()

            action = action_r_dqn(turn, positions, epsilon)
            if action is None:
                break

            # (optional) small step penalty for MDP shaping
            step_reward = -1  # keep original choice

            # apply action
            positions[action] = 1 if turn == 1 else 2
            s_next = positions.copy()

            # store transition (done=False for now)
            memory.append((s, action, step_reward, s_next, False))
            if len(memory) > batch_size:
                update_model(memory, batch_size)

            # switch turn
            turn = 2 if turn == 1 else 1

        # terminal outcome and terminal reward
        outcome = check_triplets(positions)
        if outcome == True and turn == 2:  # player1(DQN) made last move
            p1_win += 1
            terminal_reward = 2
        elif outcome == 'Tie':
            p_tie += 1
            terminal_reward = 2   # keep your original tie reward (you may want 1)
        else:
            p2_win += 1
            terminal_reward = -2

        # Add one terminal transition using last s,a to s' (already in memory)
        # Approximate by reusing the last transition's next_state == terminal s'
        # More rigor: track last s,a and use current positions as next_state.
        if len(memory) > 0:
            last_s, last_a, _, last_s_next, _ = memory[-1]
            memory.append((last_s, last_a, terminal_reward, positions.copy(), True))
            if len(memory) > batch_size:
                update_model(memory, batch_size)

        # decay epsilon safely per episode
        epsilon = max(0.05, epsilon * 0.95)

        # target network update every 500 episodes
        if p_tot % 500 == 0:
            hard_update_target()

        # reset board
        positions[:] = 0

    print('------------------')
    print('Training random finished!')
    print(f'player1 wins {((p1_win/p_tot)*100):.2f} | '
          f'player2 wins {((p2_win/p_tot)*100):.2f} | '
          f'tie {((p_tie/p_tot)*100):.2f}')