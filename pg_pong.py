
import tensorflow as tf
import numpy as np
import tflearn
import matplotlib.pyplot as plt
import time

from replay_buffer import ReplayBuffer
import gym

# ==========================
#   Training Parameters
# ==========================

# Max episode length    
MAX_EP_STEPS = 100000
# Base learning rate for the Actor network
PG_LEARNING_RATE = 1e-4

# Discount factor 
GAMMA = 0.9

MINIBATCH_SIZE = 512
SAVE_STEP = 50000
EPS_MIN = 0.0
EPS_DECAY_RATE = 0
# ===========================
#   Utility Parameters
# ===========================
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results_pg/'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000
EVAL_EPISODES = 100
RENDER = True


# ===========================
#   Actor and Critic DNNs
# ===========================
class PGNetwork(object):
    """ 
    Input to the network is the state, output is the action
    under a policy.

    """
    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim # 210*160*3
        self.a_dim = action_dim # 3: 1:hold, 2: up, 3: down

        self.learning_rate = learning_rate

        # Actor Network
        self.inputs, self.actions_out, self.log_prob = self.create_pg_network()

        self.network_params = tf.trainable_variables()

        
        # Compute the loss here 
        self.actions_in = tf.placeholder(tf.int32) 
        self.advantages = tf.placeholder(tf.float32)
        indices = tf.stack([tf.range(0, MINIBATCH_SIZE), self.actions_in], axis=1) # construct two columns of action choosing array
        act_prob = tf.gather_nd(self.log_prob, indices) # get the probability of chosed action of each element in a batch
        loss = -tf.reduce_sum(tf.multiply(act_prob, self.advantages)) # surrogate function: sum(-A(t)*log[pi(action)])

        # Optimization Op
        self.actor_global_step = tf.Variable(0, name='pg_global_step', trainable=False)

        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.optimize = self.optimizer.minimize(loss)

    def create_pg_network(self): 
        inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1], self.s_dim[2]])
        net = tflearn.conv_2d(inputs, 16, 3, activation='relu', name='conv1')
        net = tflearn.conv_2d(net, 16, 3, activation='relu', name='conv1')
        net = tflearn.conv_2d(net, 16, 3, activation='relu', name='conv1')

        net = tflearn.fully_connected(net, 128, activation='relu')
        net = tflearn.fully_connected(net, 64, activation='relu')
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        logits = tflearn.fully_connected(net, self.a_dim, activation='tanh', weights_init=w_init)

        actions_out = tf.reshape(tf.multinomial(logits, 1), []) # choose the actions based on the log probability of each element of logits

        log_prob = tf.log(tf.nn.softmax(logits))

        return inputs, actions_out, log_prob

    def train(self, states, actions_in, advantages):
        # print("actor global step: ", self.actor_global_step.eval())

        self.sess.run(self.optimize, feed_dict={
            self.inputs: states,
            self.actions_in: actions_in,
            self.advantages: advantages
        })

    def predict(self, inputs):
        return self.sess.run(self.actions_out, feed_dict={
            self.inputs: inputs
        })



# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries(): 
    reward_eps = tf.Variable(0.)
    tf.summary.scalar('Reward_episode', reward_eps)

    summary_vars = [reward_eps]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================
def train(sess, env, pg, global_step):
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())

    # load model if have
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(SUMMARY_DIR)
    
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print ("Successfully loaded:", checkpoint.model_checkpoint_path)
        print("global step: ", global_step.eval())

    else:
        print ("Could not find old network weights")

    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    i = global_step.eval()


    eval_acc_reward = 0
    tic = time.time()
    eps = 1

    while True:
        i += 1
        s = env.reset()
        ep_ave_max_q = 0
        eps *= EPS_DECAY_RATE
        eps = max(eps, EPS_MIN)

        episode_s, episode_acts, episode_rewards = [], [], []

        if i % SAVE_STEP == 0 : # save check point every 1000 episode
            sess.run(global_step.assign(i))
            save_path = saver.save(sess, SUMMARY_DIR + "model.ckpt" , global_step = global_step)
            print("Model saved in file: %s" % save_path)
            print("Successfully saved global step: ", global_step.eval())



        action = pg.predict(np.reshape(s, np.hstack((1, pg.s_dim)))) # roll the dice based on the log_probability
        action += 1 # one offset of each action: 1,2,3
        print action

        s2, r, terminal, info = env.step(action)
        if RENDER:
            env.render()
        # plt.imshow(s2, interpolation='none')
        # plt.show()
        episode_s.append(s)
        episode_acts.append(action)
        episode_rewards.append(r)

        s = s2
        eval_acc_reward += r

        if terminal: # Must be trained while terminal of one episode
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            episode_rewards = np.asarray(episode_rewards)
            # print('episode_rewards', episode_rewards)

            episode_rewards = discount_rewards(episode_rewards)
            # print('after', episode_rewards)
            # update buffer
            for n in range(len(episode_rewards)):
                replay_buffer.add(np.reshape(episode_s[n], (pg.s_dim)), episode_acts[n],
                 episode_rewards[n], terminal, np.reshape(episode_s[n], (actor.s_dim)))
            
            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:     
                s_batch, a_batch, r_batch, t_batch, _ = replay_buffer.sample_batch(MINIBATCH_SIZE)
                # Update the actor policy using the sampled gradient
                pg.train(s_batch, a_batch, r_batch)



            # print '| Reward: %.2i' % int(ep_reward), " | Episode", i, \
            #     '| Qmax: %.4f' % (ep_ave_max_q / float(j+1))

            if i%EVAL_EPISODES == 0:
                # summary
                time_gap = time.time() - tic
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: (eval_acc_reward+EVAL_EPISODES)/2,
                })
                writer.add_summary(summary_str, i)
                writer.flush()

                print ('| epi_reward: %i %%' % (eval_acc_reward), "| Episode", i, \
                     ' | Time: %.2f' %(time_gap), ' | Eps: %.2f' %(eps))
                tic = time.time()

                # print(' 100 round reward: ', eval_acc_reward)
                eval_acc_reward = 0


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    discounted_r = discounted_r.astype(np.float32)
    running_add = 0

    for t in reversed(xrange(0, r.size)):
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add

    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_r -= np.mean(discounted_r) # baseline: the mean of return value in one episode
    # discounted_r /= np.std(discounted_r + 1e-10)

    return discounted_r

def get_env(task, seed):
    env_id = task.env_id
    env = gym.make(env_id)

    return env


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
 
        global_step = tf.Variable(0, name='global_step', trainable=False)

        benchmark = gym.benchmark_spec('Atari40M')

        # Change the index to select a different game.
        task = benchmark.tasks[3]

        # Run training
        seed = 0 # Use a seed of zero (you may want to randomize the seed!)
        env = get_env(task, seed)

        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)

        # state_dim = np.prod(env.observation_space.shape)
        state_dim = env.reset().shape
        print('state_dim:',state_dim)
        action_dim = env.action_space.n/2 # 3 actions: 1: hold, 2: up, 3: down
        print('action_dim:',action_dim)


        pg = PGNetwork(sess, state_dim, action_dim, PG_LEARNING_RATE)

        train(sess, env, pg, global_step)

if __name__ == '__main__':
    tf.app.run()
