import datetime
import os
import random
import time
from collections import namedtuple
from time import sleep

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gym.envs.classic_control import rendering
from pyvirtualdisplay import Display
from tensorflow.keras.layers import Conv2D, Dense, Input
from PIL import Image

def reward_shaper(rew, done):
    return rew if not done else -10

def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0: 
        if not err: 
            print("Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l))
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)

class ReplayMemory(object):
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward, finished):
        
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = [state,action,next_state,reward,finished]
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class DQN(object):
    def __init__(self,n_actions,state_shape,learning_rate=0.00001,n_hidden_1=128,n_hidden_2=64):
        
        self.n_actions = n_actions
        self.input_shape = state_shape
        self.learning_rate = learning_rate

        self.inputs = Input(shape=self.input_shape,name='Input')
        self.x = Dense(n_hidden_1,activation='relu',name='Hidden1')(self.inputs)

        if n_hidden_2 is not None:
            self.x = Dense(n_hidden_2,activation='relu',name='Hidden2')(self.x)

        self.outputs = Dense(n_actions,name='Output')(self.x)

        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs, name='DQNetwork')

        self.loss = tf.keras.losses.Huber()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    
    def get_best_action(self,state):
        return tf.argmax(self.model(state),axis=-1)
    
    def get_best_q(self,state):
        return np.max(self.model(state),axis=-1)

class ActionScheduler(object):

    def __init__(self,
                DQNetwork,
                max_steps,
                n_actions,
                eps_start=1,
                eps_end_exploration=0.1,
                eps_end=0.01,
                exploration_steps=1000000,
                full_exploration_steps=50000):
        
        self.DQNetwork = DQNetwork
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

    def _compute_current_eps(self,step):
        if step<=self.full_exploration_steps:
            eps = self.eps_start
        elif step>self.full_exploration_steps and step<=(self.full_exploration_steps + self.exploration_steps):
            eps = step*self.slope_exploration + self.intercept
        else:
            eps = step*self.final_slope + self.intercept_2
        return eps
    
    def get_action(self,state,step):

        eps = self._compute_current_eps(step)
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.n_actions)
        return self.DQNetwork.get_best_action(state).numpy()[0]
    
    def plot_eps_schedule(self):
        eps_values = []
        for i in range(self.max_steps):
            eps_values.append(self.compute_current_eps(i))
        plt.plot(range(self.max_steps),eps_values)
        plt.show()
        pass

def train_step(policy_net,target_net,replay_memory,batch_size,gamma):
    if len(replay_memory) < batch_size:
        return

    #Transitions: state, action, next_state, reward, finished
    transitions = replay_memory.sample(3)

    states = np.array([elem[0] for elem in transitions])
    action_batch = tf.convert_to_tensor(np.array([[elem[1]] for elem in transitions]),dtype="int32")
    next_states = np.array([elem[2] for elem in transitions])
    rewards = np.array([elem[3] for elem in transitions])
    terminal_flags = np.array([int(elem[4]) for elem in transitions])

    target_q_values = target_net.get_best_q(next_states)
    target_q = rewards + (gamma*target_q_values * (1-terminal_flags))

    with tf.GradientTape() as tape:
        pred_policy = policy_net.model(states)
        pred_policy = tf.gather_nd(pred_policy, action_batch, batch_dims=1)
        loss_value = policy_net.loss(target_q,pred_policy)
        gradients = tape.gradient(loss_value, policy_net.model.trainable_variables)

    policy_net.optimizer.apply_gradients(zip(gradients, policy_net.model.trainable_variables))

    return loss_value

def test(episode, env, policy_net, show=False, save_gif = False, gif_dir = None):

    if show:
        viewer = rendering.SimpleImageViewer()
        rgb = env.render('rgb_array')

    if save_gif:
        frame = env.render('rgb_array')
        frame = repeat_upsample(frame,4,4)
        gif_frames = [frame]

    state, ep_reward, done = env.reset(), 0, False
    steps = 0
    while not done and steps<500:

        if show:
            upscaled=repeat_upsample(rgb,4, 4)
            viewer.imshow(upscaled)

        action = policy_net.get_best_action(np.expand_dims(state,axis=0)).numpy()[0]
        state, reward, done, info = env.step(action)

        reward = reward_shaper(reward,done)

        '''
        if info['ale.lives']==4:
            done = True
        '''

        ep_reward += reward
        steps += 1
        
        if show:
            rgb = env.render('rgb_array')
        
        if save_gif:
            frame = env.render('rgb_array')
            frame = repeat_upsample(frame,4,4)
            gif_frames.append(frame)
    
    if show:
        viewer.close()

    name_gif = "test_episode_" + str(episode) + ".gif"

    if save_gif:
        images = []
        for frame in gif_frames:
            images.append(Image.fromarray(frame))
        images[0].save(gif_dir+name_gif,save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)

    env.close()
    
    return ep_reward, steps


class TensorboardSummary(object):

    def __init__(self,path_logs):

        self.path_logs = path_logs
        
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = os.path.join(self.path_logs,self.current_time,"train_rewards")
        self.val_log_dir = os.path.join(self.path_logs,self.current_time,"validation_rewards")
        self.q_loss_log_dir = os.path.join(self.path_logs,self.current_time,"q_loss")

        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.validation_summary_writer = tf.summary.create_file_writer(self.val_log_dir)
        self.q_loss_summary_writer = tf.summary.create_file_writer(self.q_loss_log_dir)

        self.dict_param_writer = {'train_rewards': (self.train_summary_writer,'reward'),
                                    'validation_rewards': (self.validation_summary_writer,'reward'),
                                    'q_loss': (self.q_loss_summary_writer,'loss')}

    def update_values(self,param,value,episode):
        with self.dict_param_writer[param][0].as_default():
            tf.summary.scalar(self.dict_param_writer[param][1], value, step=episode)
