#main.py

import torch
from training.play_to_train import play_dqn
from training.play_random import play_r_dqn
from agents.human_vs_bot import play_vs_dqn 


if torch.cuda.is_available():
    print('GPU available!')
else:
    print('GPU not available!')


#play_r()

#play()
play_r_dqn()

play_dqn()


p_again = 'y'

while p_again == 'y':
    print('You are player 1! which one should start first?')
    turn = int(input('[1/2]:'))
    play_vs_dqn(turn)
    p_again = input('would you like to play again? [y/n]:')