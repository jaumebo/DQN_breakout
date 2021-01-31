import gym

env_name = "Breakout-ram-v0"
env = gym.make(env_name)
print(env.observation_space.shape[0])
rewards = 0
steps = 0
done = False
observation = env.reset()
observation = env.step(1)
env.render()
while not done:

    observation, reward, done, info = env.step(3)
    env.render()

    steps += 1
    rewards += reward

    if info['ale.lives']==4:
        done = True

print("steps: {} rewards {}: ".format(steps, rewards))