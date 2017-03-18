
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
SUMMARY_DIR = './results_dqn/rnn_dqn'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000
EVAL_EPISODES = 1000
RENDER = True
TEST_STEP = 1000000
TEST_TIMES = 1
###############
# Game Config #
###############
GAME            =  'CarRacing-v0'
ACTION_ACCEL    =  [0, 0.3, 0]
ACTION_BRAKE    =  [0, 0, 0.05]
ACTION_LEFT     =  [-1, 0, 0.1]
ACTION_RIGHT    =  [ 1, 0, 0.1]
ACTIONS         =  [ACTION_ACCEL, ACTION_LEFT, ACTION_RIGHT, ACTION_BRAKE]
# ACTIONS         =  [ACTION_ACCEL, ACTION_LEFT, ACTION_RIGHT]
ACTION_SIZE     =  len(ACTIONS)


# ===========================
#   Q DNN
# ===========================
class QNetwork(object):
    """ 
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, layers=10):
        self.sess = sess
        self.s_dim = state_dim  # 96*96*3
        self.a_dim = action_dim # 4
        self.learning_rate = learning_rate
        self.tau = tau
        self.layers = layers
        # Create the Qnet network
        self.inputs, self.out = self.create_Q_network()

        self.network_params = tf.trainable_variables()


    def create_Q_network(self):
        inputs = tflearn.input_data(shape=self.s_dim)
        features = tflearn.conv_2d(inputs, 16, 8, activation='relu', name='conv1')
        features = tflearn.conv_2d(features, 16, 8, activation='relu', name='conv2')
        # features = tflearn.layers.conv.max_pool_2d (features, 2, strides=None, padding='same', name='MaxPool2D1')
        features = tflearn.conv_2d(features, 16, 8, activation='relu', name='conv3')
        # rnn
        features_rnn = tflearn.layers.core.flatten(features)
        fc1 = tflearn.fully_connected(features_rnn, 32)
        fc2 = tflearn.fully_connected(fc1, 64)
        fc_fb = tflearn.fully_connected(fc2, 32)

        net = tflearn.activation(tf.matmul(features_rnn,fc1.W) + fc1.b, activation='relu')
        for i in range(self.layers - 1):
            net = tflearn.activation(tf.matmul(net,fc2.W) + fc2.b, activation='relu')
            net = tflearn.activation(tf.matmul(net,fc_fb.W) + tf.matmul(features_rnn, fc1.W) + fc_fb.b + fc1.b, activation='relu')
        net = tflearn.activation(tf.matmul(net,fc2.W) + fc2.b, activation='relu')

        net = tflearn.layers.normalization.batch_normalization (net, name='BatchNormalization1')
        out = tflearn.fully_connected(net, self.a_dim)

        return inputs, out

    def predict(self, inputs): # predict q values of 4 action
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
        })

# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries(): 
    eval_points = tf.Variable(0.)
    tf.summary.scalar('eval_points', eval_points)

    summary_var = [eval_points]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_var
# ===========================
#   Agent Testing
# ===========================
def Test_Agent_Once(sess, env, Qnet, global_step):
    
    s = env.reset()
    s = prepro(s)
    eval_points = 0
    terminal = True
    for i in range(TEST_STEP):
    # while terminal:
        if RENDER:
            env.render()

        predicted_q_value = Qnet.predict(np.reshape(s, np.hstack((1, Qnet.s_dim))))
        action = np.argmax(predicted_q_value)
        # print action
        exe_action = action_demask(action)

        s, r, terminal, _ = env.step(exe_action)
        s = prepro(s)
        eval_points += r

        if terminal:
            break
    return eval_points
    


def action_demask(action):
    return ACTIONS[action]

def prepro(state):
    """ prepro state to 3D tensor   """
    # print('before: ', state.shape)
    state = rgb2grey(state)
    state = state.reshape(state.shape[0], state.shape[1], 1)
    # print('after: ', state.shape)
    # plt.imshow(state, interpolation='none')
    # plt.show()
    # state = state.astype(np.float).ravel()
    return state

def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
 
        global_step = tf.Variable(0, name='global_step', trainable=False)

        env = gym.make(GAME)
        env = wrappers.Monitor(env, '/tmp/CarRacing_plain_discrete2', force=True)
        state = env.reset()

        state_dim = (state.shape[0], state.shape[1], 1)
        print('state_dim:',state_dim)
        action_dim = ACTION_SIZE
        print('action_dim:',action_dim)

        Qnet = QNetwork(sess, state_dim, action_dim, Q_LEARNING_RATE, TAU)

        summary_ops, summary_var = build_summaries()
        # load model if have
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(SUMMARY_DIR)

        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print ("Successfully loaded:", checkpoint.model_checkpoint_path)
            print("global step: ", global_step.eval())

        else:
            print ("Could not find old network weights")
            return 

        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        for i in range(TEST_TIMES):
            eval_points = Test_Agent_Once(sess, env, Qnet, global_step)
        
            summary_str = sess.run(summary_ops, feed_dict={
                        summary_var[0]: eval_points
                    })
            writer.add_summary(summary_str)
            writer.flush()
if __name__ == '__main__':
    tf.app.run()
