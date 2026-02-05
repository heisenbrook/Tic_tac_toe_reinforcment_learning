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
├── experiments/
│   ├── logs/                    # Training logs
│   └── artifacts/               # Saved models / Q-tables
│
├── requirements.txt
└── README.md

```
