from DQN_functions import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#env_name = "Breakout-ram-v0"
env_name = "CartPole-v1"
env = gym.make(env_name)

path_logs = 'logs/'

seed = 123  # random seed
log_interval = 200  # controls how often we log progress, in episodes

lr = 1e-2
gamma = 0.99 
eps_start = 1
eps_end_exploration = 0.1
eps_end = 0.0001
max_steps = 100000
exploration_steps = 80000
full_exploration_steps = 10000
batch_size = 256

target_update = 1000

num_states = env.observation_space.shape[0]
n_actions = env.action_space.n
hidden_units = [200, 200]

max_experiences = 10000
min_experiences = 100

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = path_logs + current_time

summary_writer = tf.summary.create_file_writer(log_dir)

epsilon_scheduler = EpsilonScheduler(max_steps=max_steps,
                                    n_actions=n_actions,
                                    eps_start=eps_start,
                                    eps_end_exploration=eps_end_exploration,
                                    eps_end=eps_end,
                                    exploration_steps=exploration_steps,
                                    full_exploration_steps=full_exploration_steps)

DQagent = DQN(num_states, n_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)

total_rewards = []
step = 0
episodes = 0
while step<max_steps:
    reward, losses, step = game_step(env, DQagent, epsilon_scheduler, target_update, step)
    episodes += 1
    total_rewards.append(reward)
    avg_rewards = np.asarray(total_rewards[max(0, episodes - 100):(episodes + 1)]).mean()
    with summary_writer.as_default():
        tf.summary.scalar('episode reward', reward, step=episodes)
        tf.summary.scalar('running avg reward (100 episodes)', avg_rewards, step=episodes)
        tf.summary.scalar('average loss)', losses, step=episodes)
    if episodes % 100 == 0:
        print("episode:", episodes, "episode reward:", reward, "running avg reward (100 episodes):", avg_rewards,
                "episode loss: ", losses)

print("Avg reward for last 100 episodes:", avg_rewards)
print("Total episodes: ", episodes)
make_video(env, DQagent)
env.close()
