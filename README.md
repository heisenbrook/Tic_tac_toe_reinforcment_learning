### Project - Tic Tac Toe Game with Machine learning


This project implements and compares **tabular Q-Learning** and a **Deep Q-Network (DQN)** applied to the game of Tic-Tac-Toe.  
The goal is to study reinforcement learning techniques on a simple, fully observable, discrete environment.


---

## Project Structure

The repository is organized to clearly separate agents, training logic, and execution.

``` text
tic_tac_toe_rl/
│
├── main.py
│
├── src/
│   ├── agents/
│   │   ├── q_player.py          # Tabular Q-learning agent
│   │   ├── dqn_player.py        # Deep Q-Network agent (PyTorch)
│   │   └── human_vs_bot.py      # Human vs Q/DQN interaction
│   │
│   ├── training/
│   │   ├── play_random.py       # Training against random agents
│   │   ├── play_to_train.py     # Q-learning and DQN training loops
│   │   └── play_h_vs_h.py       # Human vs human (baseline)
│   │
│   └── utils/
│       └── (auxiliary utilities)
│
├──  logs/                    # Training logs
├─ artifacts/               # Saved models / Q-tables
│
├── requirements.txt
└── README.md

```

# High‑level design


*State*: 
```
A flat vector of 9 cells [0..8]. 0=empty, 1=X (player 1), 2=O (player 2).
```

*Actions*: 
```
Indices 0..8 where the agent plays if the spot is empty.
```

*Agents*:
```

1. Q‑Learning (tabular): Dictionary Q1 maps a stringified state to a vector of 9 Q‑values. Trained via self‑play or vs random. 

2. DQN (PyTorch): A small MLP (9 → 200 → 200 → 9) producing Q‑values for each action. Uses a target network and experience replay.
```


*Training Loops*:
```
1. play_random.py: Train vs a random opponent (both Q‑table and DQN variants).

2. play_to_train.py: Self‑play training (Q‑table and DQN variants).
```


*Interaction*:
```
human_vs_bot.py: CLI loop to play human vs Q‑table or DQN.
```


*Game rules*:
```
board.py: Prints the board and checks win/tie.
```