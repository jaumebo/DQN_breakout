from DQN_functions import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Hyperparameters
env_name = "Breakout-ram-v0"

path_logs = 'logs'

seed = 123  # random seed
log_interval = 10  # controls how often we log progress, in episodes
show_eval = False  # controls if we see an evaluation episode
save_gif = True
gif_dir = "./gifs/"

gamma = 0.99 
eps_start = 1
eps_end_exploration = 0.1
eps_end = 0.01
max_steps = 3000000
exploration_steps = 1000000
full_exploration_steps = 10000
batch_size = 32

target_update = 100

### TRAIN LOOP

env = gym.make(env_name)
env.seed(seed)

state_shape = env.observation_space.shape
n_actions = env.action_space.n

policy_net = DQN(n_actions,state_shape,n_hidden_2=None)
target_net = DQN(n_actions,state_shape,n_hidden_2=None)

replay_memory = ReplayMemory(capacity=10000)

action_scheduler = ActionScheduler(DQNetwork=policy_net,
                                    max_steps=max_steps,
                                    n_actions=n_actions,
                                    eps_start=1,
                                    eps_end_exploration=eps_end_exploration,
                                    eps_end=eps_end,
                                    exploration_steps=exploration_steps,
                                    full_exploration_steps=full_exploration_steps)

tensorboard_summarizer = TensorboardSummary(path_logs)

step_count = 0
i_episode = 0
ep_rew_history = []


while step_count < max_steps:
    
    state, done = env.reset(), False
    ep_reward = 0

    while not done:

        action = action_scheduler.get_action(state,step_count)
        step_count+=1

        next_state, reward, done, info = env.step(action)

        ep_reward += reward

        if info['ale.lives']==4:
            done = True

        if done:
            next_state = state

        replay_memory.push(state,action,next_state,reward,done)

        state = next_state

        q_loss = train_step(policy_net,target_net,replay_memory,batch_size,gamma)

        if step_count % target_update == 0:
            target_net.model.set_weights(policy_net.model.get_weights())

    tensorboard_summarizer.update_values('train_rewards',ep_reward,i_episode)
    tensorboard_summarizer.update_values('q_loss',q_loss,i_episode)
    
    if i_episode % log_interval == 0 or step_count >= max_steps:
        ep_reward, val_steps = test(i_episode,env,policy_net,show=show_eval,save_gif=save_gif,gif_dir=gif_dir)
        ep_rew_history.append((i_episode, ep_reward))
        print("Episode: " + str(i_episode) + "\tTotal steps: " 
            + str(step_count) + "\tEval reward: " + str(ep_reward)
            + "\tEval steps: " + str(val_steps))

        tensorboard_summarizer.update_values('validation_rewards',ep_reward,i_episode)

    i_episode += 1






