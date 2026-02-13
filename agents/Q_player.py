# Q_player.py

"""
Update Q-table based on the state, action, reward, and new state.
Select an action using epsilon-greedy strategy based on the Q-table.
"""

#import necessary libraries
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
    
def sel_e_greedy_action(Q, positions, eps: float = 0.1):
    """
    Epsilon-greedy selection over empty cells, returns an int action.
    """
    positions = np.array(positions)
    empty_pos = np.where(positions == 0)[0]
    if len(empty_pos) == 0:
        return None

    state_key = str(positions)

    # explore
    if random.random() < eps or state_key not in Q:
        return int(np.random.choice(empty_pos))

    # exploit among empty positions
    q_val = np.array(Q[state_key])
    best_among_empty = empty_pos[np.argmax(q_val[empty_pos])]
    return int(best_among_empty)




