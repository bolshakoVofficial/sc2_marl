from smac.env import StarCraft2Env
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model

map_name = "2m_vs_2zg"
MODEL_NAME = "64x64"

# path to model file (ex: models_deltas/128x64___.h5) OR None
LOAD_MODEL = "models_deltas/64x64___444.79avg.h5"

env = StarCraft2Env(map_name=map_name)
env_info = env.get_env_info()
state_shape = env_info["state_shape"]
n_actions = env_info["n_actions"]
n_agents = env_info["n_agents"]
episode_limit = env_info["episode_limit"]
map_size = (8, 24, 25, 28)  # x1, x2, y1, y2
EPISODES = 100

# stats for matplotlib
total_steps = []
ep_reward = []
mean_total_steps = []
mean_ep_reward = []
ep_reward_agent1 = []
ep_reward_agent2 = []
mean_ep_reward_agent1 = []
mean_ep_reward_agent2 = []

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)


class DQN:
    def __init__(self):
        self.model = self.create_model()

    def create_model(self):
        model = load_model(LOAD_MODEL)
        print(f"Loaded model: {LOAD_MODEL}")
        return model

    def get_q_values(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]


def avail_joint_actions(avail_actions_array):
    """
    Takes array of shape [[agent0_avail_actions], [agent1_avail_actions]]

    Returns all available joint actions
    """

    if avail_actions_array[0][0] == 0:
        return avail_actions_array[1]

    if avail_actions_array[1][0] == 0:
        return avail_actions_array[0] * n_actions

    all_actions = []

    for agent0_act in range(len(avail_actions_array[0])):
        for agent1_act in range(len(avail_actions_array[1])):
            all_actions.append(jal_encoder(avail_actions_array[0][agent0_act],
                                           avail_actions_array[1][agent1_act]))
    return all_actions


def jal_encoder(action1, action2):
    return action1 * n_actions + action2


def jal_decoder(action):
    return [action // n_actions, action % n_actions]


network = DQN()

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episode"):
    env.reset()
    terminated = False
    episode_reward = 0
    episode_reward1 = 0
    episode_reward2 = 0
    n_steps = 1

    while not terminated:
        all_avail_actions = []
        actions = []
        rewards = []
        state = env.get_state()

        for agent_id in range(n_agents):
            avail_actions = env.get_avail_agent_actions(agent_id)
            avail_actions[1] = 0
            avail_actions_ind = np.nonzero(avail_actions)[0]
            all_avail_actions.append(avail_actions_ind)

        # joint actions that satisfy restrictions of env
        possible_joint_actions = avail_joint_actions(all_avail_actions)

        q_values = network.get_q_values(state)
        avail_qs = np.array([-100] * len(q_values))

        # get q values of available actions
        for possible_action in possible_joint_actions:
            avail_qs[possible_action] = q_values[possible_action]

        joint_action = np.argmax(avail_qs)
        actions_pair = jal_decoder(joint_action)

        # reward assigning
        for agent_id in range(n_agents):
            action = actions_pair[agent_id]
            if (action == 6) or (action == 7):
                rewards.append(10)
            else:
                rewards.append(-2)
            actions.append(action)

        _, terminated, _ = env.step(actions)

        episode_reward += sum(rewards)

        episode_reward1 += rewards[0]
        episode_reward2 += rewards[1]
        n_steps += 1

    # for matplotlib
    ep_reward.append(episode_reward)
    total_steps.append(n_steps)
    ep_reward_agent1.append(episode_reward1)
    ep_reward_agent2.append(episode_reward2)

    if not episode % 10:
        mean_ep_reward.append(np.mean(ep_reward[-10:]))
        mean_total_steps.append(np.mean(total_steps[-10:]))
        mean_ep_reward_agent1.append(np.mean(ep_reward_agent1[-10:]))
        mean_ep_reward_agent2.append(np.mean(ep_reward_agent2[-10:]))

    game_stats = env.get_stats()
    print()
    print('Episode ', episode)
    print(f"Steps: {n_steps}   Reward: {round(episode_reward, 3)}")
    print('Won: {}    Played: {}    Win rate: {}'.format(game_stats['battles_won'],
                                                         game_stats['battles_game'],
                                                         round(game_stats['win_rate'], 3)))

x = np.linspace(0, EPISODES, EPISODES // 10)

with open(f"{map_name}_{MODEL_NAME}_DQN_plot_mean_ep_reward_agent1", 'wb') as f:
    pickle.dump(mean_ep_reward_agent1, f)

with open(f"{map_name}_{MODEL_NAME}_DQN_plot_mean_ep_reward_agent2", 'wb') as f:
    pickle.dump(mean_ep_reward_agent2, f)

with open(f"{map_name}_{MODEL_NAME}_DQN_plot_mean_ep_reward", 'wb') as f:
    pickle.dump(mean_ep_reward, f)

with open(f"{map_name}_{MODEL_NAME}_DQN_plot_mean_total_steps", 'wb') as f:
    pickle.dump(mean_total_steps, f)

with open(f"{map_name}_{MODEL_NAME}_DQN_plot_x", 'wb') as f:
    pickle.dump(x, f)

plt.plot(x, mean_ep_reward_agent1, label='Agent0 reward')
plt.plot(x, mean_ep_reward_agent2, label='Agent1 reward')
plt.legend()

fig, ax = plt.subplots(1, 2)
ax1, ax2 = ax.flatten()
ax1.plot(x, mean_ep_reward)
ax1.set_title('Rewards')
ax2.plot(x, mean_total_steps)
ax2.set_title('Steps')
fig.set_size_inches(15, 4)
fig.subplots_adjust(hspace=0.2, wspace=0.2)
plt.show()

env.close()
