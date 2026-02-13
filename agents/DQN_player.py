# DQN_player.py
"""
Deep Q-Network implementation for Tic Tac Toe
This module defines a DQN architecture and associated functions 
for action selection and model updates.
"""


#import necessary libraries
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.activation = nn.ReLU()
        self.layer1 = nn.Linear(inp, 200)
        self.layer2 = nn.Linear(200, 200)
        self.layer3 = nn.Linear(200, out) 
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(next(self.parameters()).device)
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        return x
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Main Network
model = DQN(9, 9).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()

# Target Network
target_model = DQN(9, 9).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

def hard_update_target():
    """
    Update the target network by copying the weights from the main model. 
    Call it periodically from the training loop.
    """
    target_model.load_state_dict(model.state_dict())
    target_model.eval()


def available_actions(state):
    s = np.array(state)
    return np.where(s == 0)[0]  # integer indices


def select_action(state, epsilon, mask_illegal=True):
    """
    Epsilon-greedy action selection constrained to empty cells.
    Returns an int action in [0..8].
    """
    avail = available_actions(state)
    if len(avail) == 0:
        return None  # board full

    if np.random.rand() < epsilon:
        return int(np.random.choice(avail))

    with torch.no_grad():
        q = model(torch.tensor(state, dtype=torch.float32).to(device))
        if q.ndim > 1:  # ensure 1D
            q = q.squeeze(0)
        if mask_illegal:
            mask = torch.full_like(q, float('-inf'))
            mask[torch.tensor(avail, device=q.device)] = 0.0
            q = q + mask
        return int(torch.argmax(q).item())

def update_model(memory, batch_size, gamma=0.7):
    """
    Sample from replay memory and do one-step TD update.
    """
    exp = random.sample(memory, batch_size)

    for state, action, reward, next_state, done in exp:
        # Prepare tensors
        state_t = torch.tensor(state, dtype=torch.float32).to(device)
        q_current = model(state_t)

        # Compute target for the taken action only
        target_vec = q_current.clone().detach()

        if done:
            target_value = reward
        else:
            with torch.no_grad():
                next_state_t = torch.tensor(next_state, dtype=torch.float32).to(device)
                q_next = target_model(next_state_t)
                if q_next.ndim > 1:
                    q_next = q_next.squeeze(0)
                # mask illegal actions in next_state
                avail_next = np.where(np.array(next_state) == 0)[0]
                if len(avail_next) == 0:
                    max_next = 0.0
                else:
                    mask = torch.full_like(q_next, float('-inf'))
                    mask[torch.tensor(avail_next, device=q_next.device)] = 0.0
                    max_next = torch.max(q_next + mask).item()
                target_value = reward + gamma * max_next

        a = int(action)
        target_vec[a] = target_value

        optimizer.zero_grad()
        loss = loss_fn(model(state_t), target_vec)
        loss.backward()
        optimizer.step()

def save_model(path:str):
    torch.save(model.state_dict(), path)

def load_model(path:str):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()