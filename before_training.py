
import tensorflow as tf

import numpy as np
import tflearn
# import matplotlib.pyplot as plt
import time

from replay_buffer_dqn import ReplayBuffer
import gym
from gym import wrappers
from skimage.color import rgb2grey
# ==========================
#   Training Parameters
# ==========================

# Max episode length    
MAX_EP_STEPS = 1000

# Base learning rate for the Qnet Network
Q_LEARNING_RATE = 1e-3
# Discount factor 
GAMMA = 0.9
# Soft target update param
TAU = 0.001
TARGET_UPDATE_STEP = 100

MINIBATCH_SIZE = 32
SAVE_STEP = 100
EPS_MIN = 0.05
EPS_DECAY_RATE = 0.999
EPS_UPDATE = 20
# ===========================
#   Utility Parameters
# ===========================
# map size
MAP_SIZE  = 8
PROBABILITY = 0.1
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results_dqn/dqn_306'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000
EVAL_EPISODES = 1000
RENDER = True
TEST_STEP = 1000
TEST_TIMES = 1
###############
# Game Config #
###############
GAME            =  'CarRacing-v0'
ACTION_ACCEL    =  [0, 0.5, 0]
# ACTION_BRAKE    =  [0, 0, 0.05]
ACTION_LEFT     =  [-1, 0, 0.02]
ACTION_RIGHT    =  [ 1, 0, 0.02]
# ACTIONS         =  [ACTION_ACCEL, ACTION_LEFT, ACTION_RIGHT, ACTION_BRAKE]
ACTIONS         =  [ACTION_ACCEL, ACTION_LEFT, ACTION_RIGHT]
ACTION_SIZE     =  len(ACTIONS)
  


def action_demask(action):
    return ACTIONS[action]

def main(_):
    
    env = gym.make(GAME)
    env = wrappers.Monitor(env, '/tmp/CarRacing_plain_discrete2', force=True)
    state = env.reset()

    action_dim = ACTION_SIZE
    print('action_dim:',action_dim)

    terminal = True
    for i in range(TEST_STEP):
    # while terminal:
        if RENDER:
            env.render()

        action = np.random.randint(action_dim)
        # print action
        exe_action = action_demask(action)

        s, r, terminal, _ = env.step(exe_action)

        if terminal:
            break

        
if __name__ == '__main__':
    tf.app.run()
