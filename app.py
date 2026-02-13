# todo: mettere utilities in utils.py, agent loading in src/agents/__init__.py, etc. per pulizia


from flask import Flask, render_template, request, jsonify, session
import os
import numpy as np

from src.board import check_triplets
from agents.Q_player import sel_e_greedy_action
from agents.DQN_player import select_action, load_model

# ---------- Config ----------
DQN_WEIGHTS = "artifacts/dqn.pth"
QTABLE_TXT  = "artifacts/Q1.txt"
SECRET_KEY   = "change-me-please"  #todo: set via env in prod

app = Flask(__name__)
app.secret_key = SECRET_KEY

# Globals
Qtable = {}
DQN_READY = False

# ---------- Utilities ----------
def load_qtable(path):
    if not os.path.exists(path):
        return {}
    Q = {}
    with open(path, "r") as f:
        for line in f:
            if "\t" not in line:
                continue
            key, val = line.strip().split("\t", 1)
            try:
                Q[key] = np.array(eval(val))
            except Exception:
                pass
    return Q

def init_session():
    session['board'] = [0]*9      # 0 empty, 1 X (human), 2 O (bot)
    session['turn'] = 1           # 1=human, 2=bot
    session['agent'] = 'DQN'      # 'DQN' or 'Q-Learning'
    session['done'] = False
    session['msg'] = "Your turn (X)!"

def readable_board(board):
    m = {0: " ", 1: "X", 2: "O"}
    return [m[v] for v in board]

def bot_move():
    if session.get('done'):
        return False

    board = np.array(session['board'], dtype=int)
    agent = session.get('agent', 'DQN')

    if agent == 'DQN' and DQN_READY:
        a = select_action(board, epsilon=0.0)
    elif agent == 'Q-Learning':
        a = sel_e_greedy_action(Qtable, board, eps=0.0)
    else:
        # fallback random if no models
        empties = np.where(board == 0)[0]
        if len(empties) == 0:
            a = None
        else:
            a = int(np.random.choice(empties))

    if a is None or board[a] != 0:
        empties = np.where(board == 0)[0]
        if len(empties) == 0:
            return False
        a = int(np.random.choice(empties))

    board[a] = 2
    session['board'] = board.tolist()
    return True

def check_end_and_message():
    board = np.array(session['board'], dtype=int)
    outcome = check_triplets(board)
    if outcome == True:
        # Winner is the player who just moved (turn flipped after each move).
        winner = 1 if session['turn'] == 2 else 2
        session['msg'] = "You win! (X)" if winner == 1 else "Bot wins! (O)"
        session['done'] = True
        return True
    elif outcome == "Tie":
        session['msg'] = "Draw!"
        session['done'] = True
        return True
    return False

# ---------- Routes ----------
@app.before_request
def ensure_session():
    if 'board' not in session:
        init_session()

@app.route("/", methods=["GET"])
def index():
    board = session['board']
    return render_template("index.html",
                           board=readable_board(board),
                           raw_board=board,
                           agent=session['agent'],
                           msg=session['msg'],
                           done=session['done'])

@app.route("/set_agent", methods=["POST"])
def set_agent():
    agent = request.form.get("agent", "DQN")
    session['agent'] = agent
    session['msg'] = f"Playing vs {agent}. Your turn (X)!"
    return jsonify(success=True, agent=agent)

@app.route("/reset", methods=["POST"])
def reset():
    init_session()
    return jsonify(success=True)

@app.route("/move", methods=["POST"])
def move():
    if session.get('done'):
        return jsonify(success=False, msg=session['msg'])

    try:
        idx = int(request.form.get("idx"))
    except (TypeError, ValueError):
        return jsonify(success=False, msg="Invalid move index")

    if not (0 <= idx <= 8):
        return jsonify(success=False, msg="Index out of range")

    board = np.array(session['board'], dtype=int)
    if board[idx] != 0:
        return jsonify(success=False, msg="Cell already occupied")

    # Human move
    board[idx] = 1
    session['board'] = board.tolist()
    session['turn'] = 2

    if check_end_and_message():
        return jsonify(success=True, board=readable_board(session['board']),
                       raw=session['board'], msg=session['msg'], done=True)

    # Bot move
    bot_move()
    session['turn'] = 1

    if check_end_and_message():
        return jsonify(success=True, board=readable_board(session['board']),
                       raw=session['board'], msg=session['msg'], done=True)

    return jsonify(success=True, board=readable_board(session['board']),
                   raw=session['board'], msg="Your turn (X)!", done=False)

# ---------- Startup ----------
if __name__ == "__main__":
    # Try to load models
#    global Qtable, DQN_READY
    Qtable = load_qtable(QTABLE_TXT)
    DQN_READY = False
    if os.path.exists(DQN_WEIGHTS):
        try:
            load_model(DQN_WEIGHTS)
            DQN_READY = True
            print("DQN weights loaded.")
        except Exception as e:
            print(f"Could not load DQN weights: {e}")

    app.run(host="0.0.0.0", port=5000, debug=True)
