import datetime
import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gym import wrappers


class NN_Model(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(NN_Model, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='tanh', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output

class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.policy_net = NN_Model(num_states, hidden_units, num_actions)
        self.target_net = NN_Model(num_states, hidden_units, num_actions)
        self.memory= {'state': [], 'action': [], 'reward': [], 'next_state': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.loss = tf.keras.losses.Huber()

    def predict_policy(self, inputs):
        return self.policy_net(np.atleast_2d(inputs.astype('float32')))

    def predict_target(self, inputs):
        return self.target_net(np.atleast_2d(inputs.astype('float32')))

    def train_step(self):
        if len(self.memory['state']) < self.min_experiences:
            return 0
        memory_ids = np.random.randint(low=0, high=len(self.memory['state']), size=self.batch_size)

        states = np.asarray([self.memory['state'][i] for i in memory_ids])
        actions = np.asarray([self.memory['action'][i] for i in memory_ids])
        rewards = np.asarray([self.memory['reward'][i] for i in memory_ids])
        next_states = np.asarray([self.memory['next_state'][i] for i in memory_ids])
        dones = np.asarray([self.memory['done'][i] for i in memory_ids])
        
        value_next = np.max(self.predict_target(next_states), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape() as tape:
            predicted_values = tf.math.reduce_sum(
                self.predict_policy(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = self.loss(actual_values,predicted_values)
        variables = self.policy_net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict_policy(states)[0])

    def add_experience(self, exp):
        if len(self.memory['state']) >= self.max_experiences:
            for key in self.memory.keys():
                self.memory[key].pop(0)
        for key, value in exp.items():
            self.memory[key].append(value)

    def update_weights(self):
        variables1 = self.target_net.trainable_variables
        variables2 = self.policy_net.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

def reward_shaper(rew, done):
    #return rew if not done else -200
    return rew

def game_step(env, DQagent, epsilon_scheduler, target_update, step):
    rewards = 0
    done = False
    observations = env.reset()
    losses = list()
    while not done:
        action = DQagent.get_action(observations, epsilon_scheduler.get_eps(step))
        prev_observations = observations
        observations, reward, done, info = env.step(action)

        if info['ale.lives']==4:
            done = True

        rewards += reward

        reward = reward_shaper(reward,done)

        exp = {'state': prev_observations, 'action': action, 'reward': reward, 'next_state': observations, 'done': done}
        DQagent.add_experience(exp)

        loss = DQagent.train_step()

        if isinstance(loss, int):
            losses.append(loss)
        else:
            losses.append(loss.numpy())

        step += 1
        if step % target_update == 0:
            DQagent.update_weights()

    return rewards, np.mean(losses), step

def make_video(env, DQagent, video_dir, video_name, max_video_steps):
    env = wrappers.Monitor(env, os.path.join(video_dir, video_name), force=True)
    rewards = 0
    steps = 0
    done = False
    observation = env.reset()
    while not done:
        action = DQagent.get_action(observation, 0)
        observation, reward, done, info = env.step(action)

        steps += 1
        rewards += reward

        if info['ale.lives']==4:
            done = True

        if rewards < 10 and steps>=max_video_steps:
            done = True



    print("Testing steps: {} rewards {}: ".format(steps, rewards))


class EpsilonScheduler(object):

    def __init__(self,
                max_steps,
                n_actions,
                eps_start=1,
                eps_end_exploration=0.1,
                eps_end=0.01,
                exploration_steps=1000000,
                full_exploration_steps=50000):
        
        self.max_steps = max_steps
        self.n_actions = n_actions

        self.eps_start = eps_start
        self.eps_end_exploration = eps_end_exploration
        self.eps_end = eps_end
        self.exploration_steps = exploration_steps
        self.full_exploration_steps = full_exploration_steps
        self.no_exploration_steps = self.max_steps-self.full_exploration_steps-self.exploration_steps

        self.slope_exploration = -(self.eps_start-self.eps_end_exploration)/self.exploration_steps
        self.intercept = self.eps_start - self.slope_exploration*self.full_exploration_steps
        self.final_slope = -(self.eps_end_exploration-self.eps_end)/(self.no_exploration_steps)
        self.intercept_2 = self.eps_end - self.final_slope*self.max_steps

    def get_eps(self,step):
        if step<=self.full_exploration_steps:
            eps = self.eps_start
        elif step>self.full_exploration_steps and step<=(self.full_exploration_steps + self.exploration_steps):
            eps = step*self.slope_exploration + self.intercept
        else:
            eps = step*self.final_slope + self.intercept_2
        return eps
    
    def plot_eps_schedule(self):
        eps_values = []
        for i in range(self.max_steps):
            eps_values.append(self.compute_current_eps(i))
        plt.plot(range(self.max_steps),eps_values)
        plt.show()
        pass
