# -------------------------------
# DGN for OpenAI Gym
# Author: Noah Golmant
# -------------------------------

import gym
import tensorflow as tf 
import numpy as np 
import random
from collections import deque

# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q 
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch
INITIAL_BETA=.00000001

#POSSIBLE_BETA_INCS = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.00015]
#POSSIBLE_BETA_INCS = [, 0.1, 1.0, 10.0, 100.0]

BETA_SCHEDULE_INC = 0.00001  # to tune

class DQN():
    # DQN Agent
    def __init__(self, session, env, beta_schedule_inc):
        self.session = session
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.beta = INITIAL_BETA
        self.beta_tensor = tf.Variable(INITIAL_BETA, dtype=tf.float32, name='beta_{}'.format(beta_schedule_inc), trainable=False)
        self.beta_inc = beta_schedule_inc
        self.increment_beta_step = tf.assign(self.beta_tensor, self.beta_tensor + beta_schedule_inc)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n 

        # init prior action distribution (uniform)
        self.prior = tf.constant(1./self.action_dim, dtype='float32', shape=[self.action_dim,])

        self.create_Q_network()
        self.create_training_method()

        # loading networks
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        #checkpoint = None
        if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.session, checkpoint.model_checkpoint_path)
                print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
                print "Could not find old network weights"

        global summary_writer
        summary_writer = tf.train.SummaryWriter('~/logs',graph=self.session.graph)

    def create_Q_network(self):
        # network weights
        W1 = self.weight_variable([self.state_dim,20])
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20,self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        # input layer
        self.state_input = tf.placeholder('float32', [None,self.state_dim])
        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
        # Q Value layer
        self.Q_value = tf.matmul(h_layer,W2) + b2
        prod = tf.exp(-self.beta_tensor * self.Q_value)
        
        self.target = tf.log(tf.matmul(prod, self.prior[:, None]))[:, 0]

    def create_training_method(self):
        self.action_input = tf.placeholder('float32', [None,self.action_dim]) # one hot presentation
        self.y_input = tf.placeholder('float32', [None])
        Q_action = tf.reduce_sum(tf.mul(self.Q_value,self.action_input),reduction_indices = 1)

        standard_cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.cost = standard_cost #+ kl_cost
        tf.scalar_summary("loss",self.cost)
        global merged_summary_op
        merged_summary_op = tf.merge_all_summaries()
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)


    def perceive(self,state,action,reward,next_state,done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1
        self.beta += self.beta_inc
        self.session.run(self.increment_beta_step)
        
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []

        #Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
        G_target_batch = self.target.eval(feed_dict={self.state_input: next_state_batch})
        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(-reward_batch[i])
            else :
                y_batch.append(-reward_batch[i] - (GAMMA / self.beta) * G_target_batch[i])

        self.optimizer.run(feed_dict={
            self.y_input:y_batch,
            self.action_input:action_batch,
            self.state_input:state_batch
            })
        summary_str = self.session.run(merged_summary_op,feed_dict={
                self.y_input : y_batch,
                self.action_input : action_batch,
                self.state_input : state_batch
                })
        summary_writer.add_summary(summary_str,self.time_step)

        # save network every 1000 iteration
        if self.time_step % 1000 == 0:
            self.saver.save(self.session, 'saved_networks/' + 'network' + '-dgn', global_step = self.time_step)

    def egreedy_action(self,state):
        Q_value = self.Q_value.eval(feed_dict = {
            self.state_input:[state]
            })[0]
        if random.random() <= self.epsilon:
            return random.randint(0,self.action_dim - 1)
        else:
            return np.argmin(Q_value)

        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000

    def action(self,state):
        return np.argmin(self.Q_value.eval(feed_dict = {
            self.state_input:[state]
            })[0])

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 1800 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode
#NUM_TESTS = 20 # number of average reward evaluations to run
TEST_PERIOD = 50

avg_rewards = []

def main():
    # initialize OpenAI Gym env and dqn agent
    # Init session
    
    session = tf.InteractiveSession()
    env = gym.make(ENV_NAME)
    agent = DQN(session, env, BETA_SCHEDULE_INC)
    
    session.run(tf.initialize_all_variables())

    for episode in xrange(EPISODE):
        # initialize task
        state = env.reset()
        # Train 
        for step in xrange(STEP):
            action = agent.egreedy_action(state) # e-greedy action for train
            next_state,reward,done,_ = env.step(action)
            # Define reward for agent
            reward_agent = -1 if done else 0.1
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break
        # Test every TEST_PERIOD episodes
        if episode % TEST_PERIOD == 0:
            total_reward = 0
            for i in xrange(TEST):
                state = env.reset()
                for j in xrange(STEP):
                    env.render()
                    action = agent.action(state) # direct action for test
                    state,reward,done,_ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            avg_rewards.append(ave_reward)
            print 'episode: ',episode,'Evaluation Average Reward:',ave_reward
    
    import csv
    f = open('dgn_rewards.csv', 'wb')
    wr = csv.writer(f)
    wr.writerow(avg_rewards)
    
    tf.reset_default_graph()
    
    # save results for uploading
    env.monitor.start('gym_results/CartPole-v0-experiment-1',force = True)
    for i in xrange(100):
        state = env.reset()
        for j in xrange(200):
            env.render()
            action = agent.action(state) # direct action for test
            state,reward,done,_ = env.step(action)
            total_reward += reward
            if done:
                break
    env.monitor.close()
    session.close()
        

if __name__ == '__main__':
    main()
