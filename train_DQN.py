from DQN_functions import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

env_name = "Breakout-ram-v0"
#env_name = "CartPole-v1"
env = gym.make(env_name)


seed = 123  # random seed
video_interval = 5000  # controls how often we create a video, in episodes
log_interval = 100 # controls how often we log progress, in episodes
model_checkpoint = 10000 # controls how often we save the weights of the trained model
max_video_steps = 1000 # controls the maximum of steps recorded in a video (applied if rewards is below 10)

lr = 1e-4
gamma = 0.99 
eps_start = 1
eps_end_exploration = 0.1
eps_end = 0.01
max_steps = 30000000
exploration_steps = 10000000
full_exploration_steps = 50000
batch_size = 64
skip_frames = 8

target_update = 10000

num_states = env.observation_space.shape[0]
n_actions = env.action_space.n - 1
hidden_units = [128, 128, 128, 128]

max_experiences = 1000000
min_experiences = 1000

path_logs = 'breakout_ram_logs/'
path_videos = 'breakout_ram_videos/'
path_checkpoints = 'breakout_ram_checkpoints/'

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = path_logs + current_time
video_dir = path_videos + current_time
weights_directory = path_checkpoints + current_time + "/"

summary_writer = tf.summary.create_file_writer(log_dir)

epsilon_scheduler = EpsilonScheduler(max_steps=max_steps,
                                    n_actions=n_actions,
                                    eps_start=eps_start,
                                    eps_end_exploration=eps_end_exploration,
                                    eps_end=eps_end,
                                    exploration_steps=exploration_steps,
                                    full_exploration_steps=full_exploration_steps)

DQagent = DQN(num_states, n_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)

if True:
    path_weights = "/Users/jaumebrossaordonez/Documents/DRL/DQN_breakout/breakout_ram_checkpoints/20200516-172356/checkpoint_episode_36000/model_checkpoint"
    DQagent.policy_net.load_weights(path_weights)
    DQagent.target_net.load_weights(path_weights)
    step = 250000
else:
    step = 0

episodes = 0
total_rewards = []

while step<max_steps:
    reward, losses, step = game_step(env, DQagent, epsilon_scheduler, target_update, step, skip_frames)
    episodes += 1
    total_rewards.append(reward)
    avg_rewards = np.asarray(total_rewards[max(0, episodes - 100):(episodes + 1)]).mean()
    with summary_writer.as_default():
        tf.summary.scalar('episode reward', reward, step=episodes)
        tf.summary.scalar('running avg reward (100 episodes)', avg_rewards, step=episodes)
        tf.summary.scalar('average loss)', losses, step=episodes)
    if episodes % log_interval == 0:
        print("episode:", episodes, "episode reward:", reward, "avg reward:", avg_rewards,
                "steps:", step, "epsilon:", epsilon_scheduler.get_eps(step))
    if episodes % video_interval == 0:
        print("Making video for episode:", episodes)
        make_video(env,DQagent,video_dir,"episode_" + str(episodes),max_video_steps,skip_frames)
    if episodes % model_checkpoint == 0:
        print("Saving model weights on checkpoint folder.")
        DQagent.policy_net.save_weights(weights_directory + "checkpoint_episode_" + str(episodes) + "/model_checkpoint")

print("Avg reward for last 100 episodes:", avg_rewards)
print("Total episodes: ", episodes)
make_video(env, DQagent, video_dir, "final_video", np.inf, skip_frames)
env.close()
