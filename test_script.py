from DQN_functions import *

'''
sample_state = np.array([[1,2,3,4],[2,3,4,5]],dtype='uint8')
entry_data = sample_state

print(entry_data.shape)
print(sample_state[0].shape)

DQNetwork = DQN(3,sample_state[0].shape)

print(DQNetwork.model.predict(entry_data))
print(DQNetwork.model(entry_data))
print(DQNetwork.get_best_action(entry_data).numpy())
'''


#scheduler = ActionScheduler(0,2000,4,exploration_steps=900,full_exploration_steps=100)
#scheduler.plot_eps_schedule()

sample_state1 = np.array([1,2,3,4],dtype='uint8')
sample_state2 = np.array([1,2,2,4],dtype='uint8')
policy_net = DQN(4,sample_state1.shape)
target_net = DQN(4,sample_state1.shape)
replay_memory = ReplayMemory(capacity=3)

replay_memory.push(sample_state1, 1, sample_state1, 1, False)
replay_memory.push(sample_state2, 1, sample_state2, 1, True)

train_step(policy_net,target_net,replay_memory,2,1)



