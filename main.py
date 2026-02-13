#!/usr/bin/env python3
import argparse
import os
import sys
import shutil


ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# GPU info
try:
    import torch
    CUDA_MSG = "GPU available!" if torch.cuda.is_available() else "GPU not available!"
except Exception:
    CUDA_MSG = "PyTorch not found or failed to import."

# Project imports
from training.play_random import play_r as train_q_vs_random
from training.play_random import play_r_dqn as train_dqn_vs_random
from training.play_to_train import play as train_q_self
from training.play_to_train import play_dqn as train_dqn_self

from agents.human_vs_bot import play_vs_dqn
from training.play_h_vs_h import play_h

from agents.DQN_player import save_model as dqn_save_model

ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
LOGS_DIR = os.path.join(ROOT, "logs")

def ensure_dirs():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

def move_qtable_to_artifacts():
    """
    Your Q self-play script writes 'Q1.txt' in CWD.
    Move it to artifacts/Q1.txt for the UI to pick up.
    """
    src_path = os.path.join(ROOT, "Q1.txt")
    dst_path = os.path.join(ARTIFACTS_DIR, "Q1.txt")
    if os.path.exists(src_path):
        ensure_dirs()
        shutil.move(src_path, dst_path)
        print(f"[INFO] Q-table moved to {dst_path}")

def save_dqn_weights():
    ensure_dirs()
    out_path = os.path.join(ARTIFACTS_DIR, "dqn.pth")
    dqn_save_model(out_path)
    print(f"[INFO] DQN weights saved to {out_path}")

def cmd_train_q_random(_args):
    print(f"[INFO] {CUDA_MSG}")
    train_q_vs_random()
    move_qtable_to_artifacts()
    print("[DONE] Q-learning vs random training completed.")

def cmd_train_q_self(_args):
    print(f"[INFO] {CUDA_MSG}")
    train_q_self()
    move_qtable_to_artifacts()
    print("[DONE] Q-learning self-play training completed.")

def cmd_train_dqn_random(_args):
    print(f"[INFO] {CUDA_MSG}")
    train_dqn_vs_random()
    save_dqn_weights()
    print("[DONE] DQN vs random training completed.")

def cmd_train_dqn_self(_args):
    print(f"[INFO] {CUDA_MSG}")
    train_dqn_self()
    save_dqn_weights()
    print("[DONE] DQN self-play training completed.")

def cmd_play_cli_vs_dqn(_args):
    print(f"[INFO] {CUDA_MSG}")
    print("You are player 1 (X). Who starts first?")
    try:
        turn = int(input("[1/2]: "))
    except Exception:
        turn = 1
    play_vs_dqn(turn)

def cmd_play_cli_hvh(_args):
    print("Human vs Human (terminal)... Who starts first?")
    try:
        turn = int(input("[1/2]: "))
    except Exception:
        turn = 1
    play_h(turn)

def cmd_server(_args):
    """
    Start the Flask app (app.py). Import then run.
    """
    print(f"[INFO] {CUDA_MSG}")
    ensure_dirs()
    try:
        from . import app as flask_app
    except Exception as e:
        print(f"[ERROR] Failed to import Flask app: {e}")
        sys.exit(1)
    flask_app.run(host="0.0.0.0", port=5000, debug=False)

def build_parser():
    p = argparse.ArgumentParser(description="Tic-Tac-Toe RL — trainer & web UI entrypoint")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("server", help="Run Flask + HTML UI").set_defaults(func=cmd_server)

    sub.add_parser("train_q_random", help="Train Q-table vs random").set_defaults(func=cmd_train_q_random)
    sub.add_parser("train_q_self", help="Train Q-table with self-play").set_defaults(func=cmd_train_q_self)

    sub.add_parser("train_dqn_random", help="Train DQN vs random").set_defaults(func=cmd_train_dqn_random)
    sub.add_parser("train_dqn_self", help="Train DQN with self-play").set_defaults(func=cmd_train_dqn_self)

    sub.add_parser("play_cli_vs_dqn", help="Play in terminal vs DQN").set_defaults(func=cmd_play_cli_vs_dqn)
    sub.add_parser("play_cli_hvh", help="Play in terminal human vs human").set_defaults(func=cmd_play_cli_hvh)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()