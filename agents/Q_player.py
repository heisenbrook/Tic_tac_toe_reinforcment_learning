# Q_player.py

import random
import numpy as np

def update_q_table(Q, state, action, reward, new_state):
    alpha = 0.1
    gamma = 1
    state = str(state)
    new_state = str(new_state)
    q_values = Q.get(state, np.zeros(9))
    next_q_values = Q.get(new_state, np.zeros(9))
    max_next_q_value = max(next_q_values)
    q_values[action] = (1 -alpha) * q_values[action] + alpha * (reward + gamma * max_next_q_value)

    Q[state] = q_values
    
def sel_e_greedy_action(Q, positions):
    str_positions = str(positions)
    empty_pos = np.argwhere(positions == 0)
    if str_positions not in Q:
        action = random.choice(empty_pos)
        return action
    else:
        q_val = Q[str_positions]
        empty_q_val = [q_val[x[0]] for x in empty_pos]
        max_q_val = np.max(empty_q_val)
        max_q_ind = random.choice(np.argwhere(empty_q_val == max_q_val))
        action = empty_pos[max_q_ind] 
        return action





