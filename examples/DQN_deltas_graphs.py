import pickle
import matplotlib.pyplot as plt

map_name = "2m_vs_2zg"
MODEL_NAME = "64x64"

# with open(f"{map_name}_{MODEL_NAME}_DQN_plot_mean_ep_reward_agent1", 'rb') as f:
#     mean_ep_reward_agent1 = pickle.load(f)
#
# with open(f"{map_name}_{MODEL_NAME}_DQN_plot_mean_ep_reward_agent2", 'rb') as f:
#     mean_ep_reward_agent2 = pickle.load(f)

with open(f"{map_name}_{MODEL_NAME}_DQN_plot_epsilon_values", 'rb') as f:
    epsilon_values = pickle.load(f)

with open(f"{map_name}_{MODEL_NAME}_DQN_plot_mean_ep_reward", 'rb') as f:
    mean_ep_reward = pickle.load(f)

with open(f"{map_name}_{MODEL_NAME}_DQN_plot_mean_total_steps", 'rb') as f:
    mean_total_steps = pickle.load(f)

with open(f"{map_name}_{MODEL_NAME}_DQN_plot_x", 'rb') as f:
    x = pickle.load(f)

# plt.plot(x, mean_ep_reward_agent1, label='Agent0 reward')
# plt.plot(x, mean_ep_reward_agent2, label='Agent1 reward')

fig, ax = plt.subplots(1, 3)
ax1, ax2, ax3 = ax.flatten()

ax1.plot(x, epsilon_values)
ax1.set_title('Epsilon')
ax2.plot(x, mean_ep_reward)
ax2.set_title('Rewards')
ax3.plot(x, mean_total_steps)
ax3.set_title('Steps')
plt.legend()
fig.set_size_inches(15, 4)
fig.subplots_adjust(hspace=0.2, wspace=0.2)
plt.show()
