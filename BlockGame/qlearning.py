# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 20:34:21 2018

@author: Lenovo

使用Q-learning玩一个小游戏
博客链接：https://blog.csdn.net/sinat_30665603/article/details/80541452
"""
# In[]
import numpy as np
import pandas as pd
import time

np.random.seed(2)
N_STATES = 8
ACTIONS = ['n', 'e', 's', 'w']
EPSILON = 0.9       # epsilon greedy
ALPHA = 0.1         #learning rate
LAMBDA = 0.9        #discount factor
MAX_EPISODES = 50
FRESH_TIME = 0.0

def build_q_table(n_states, actions):
    table = pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions)
    return table

def update_env(S, episode, step_cnt):
    env_list = ['-'] * 5
    background_list = ['x', ',', ':', ',', 'x']
    if type(S) == type("") and S.startswith('terminal'):
        S = int(S.replace('terminal', ''))
        print('Episode %d, steps %d' % (episode + 1, step_cnt));
        if S == 6:
            background_list[0] = 'o'
        elif S == 7:
            background_list[2] = 'o'
        else:
            background_list[4] = 'o'
        print('\r{}'.format(''.join(background_list)), end='')
        if S != 7:
            print('\tDead!')
        else:
            print('\tGet it!')
        time.sleep(1)

    else:
        env_list[S] = 'o'
        print(''.join(env_list))
        print(''.join(background_list))
        time.sleep(FRESH_TIME)

def choose_action(state, q_table):
    state_actions =  q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action = np.random.choice(ACTIONS)
    else:
        action = state_actions.argmax()
    return action

def get_env_feedback(S, A):
    S += 1
    if S in [1, 3, 5]:
        if A == 's':
            S_ = "terminal"
            if S == 1:
                S_ += '6'
                R = -1
            elif S == 3:
                S_ += '7'
                R = 1
            else:
                S_ += '8'
                R = -1
        elif A == 'n':
            S_ = S
            R = 0
        elif A == "e":
            if S in [1, 3]:
                S_ = S + 1
                R = 0
            else:
                S_ = S
                R = 0
        elif A == "w":
            if S in [3, 5]:
                S_ = S - 1
                R = 0
            else:
                S_ = S
                R = 0
    else:
        if A in ['n', 's']:
            S_ = S
            R = 0
        elif A == 'w':
            S_ = S - 1
            R = 0
        else:
            S_ = S + 1
            R = 0
    if type(S_) != type(''):
        S_ -= 1
    return S_, R
                
def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_cnt = 0
        S = np.random.choice([0, 1, 2, 3, 4])
        is_terminated = False
        update_env(S, episode, step_cnt)
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)

            if type(S_) != type(""):
                q_target = R + LAMBDA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True
            
            q_table.ix[S, A] += ALPHA * (q_target - q_table.ix[S, A])
            S = S_
            update_env(S, episode, step_cnt)
            step_cnt += 1
            
    return q_table

q_table = rl()
print('\r\nQ-TABLE')
print(q_table)
        