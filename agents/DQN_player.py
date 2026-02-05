#dqn_player.py

# Deep Q-Network implementation for Tic Tac Toe

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


def select_action(state, epsilon):

    if np.random.rand() < epsilon:
        return int(random.choice(state))
    else:
        q_val = model(torch.tensor(state, dtype=torch.float32).to(device))
        return torch.argmax(q_val).item()

def update_model(memory, batch_size, gamma = 0.7):
    """
    Update the DQN model using experiences sampled from the replay memory.
    """
    exp = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in exp:
        # Compute target Q value
        if done:
            target = reward
        else:
            with torch.no_grad():
                next_q = target_model(torch.tensor(next_state, dtype=torch.float32).to(device))
                target = reward + gamma * torch.max(next_q).item()

        # Current Q value 
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        q_current = model(state_tensor)

        target_vec = q_current.clone().detach() 
        # Update only the taken action
        a = int(action) if not isinstance(action, (np.ndarray, list)) else int(np.array(action).item())
        target_vec[a] = target

        optimizer.zero_grad()
        loss = loss_fn(model(state_tensor), target_vec)
        loss.backward()
        optimizer.step()