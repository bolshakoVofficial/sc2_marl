from smac.env import StarCraft2Env
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm


def choose_action(state, avail_actions_ind):
    if random.uniform(0, 1) < (1 - epsilon):
        action = np.random.choice(avail_actions_ind)
    else:
        qt_arr = np.zeros(len(avail_actions_ind))
        keys = np.arange(len(avail_actions_ind))
        act_ind_decode = dict(zip(keys, avail_actions_ind))

        for act_ind in range(len(avail_actions_ind)):
            qt_arr[act_ind] = q_table[state, act_ind_decode[act_ind]]

        action = act_ind_decode[np.argmax(qt_arr)]
    return action


def learn(state, state2, reward, action):
    q_table[state, action] = q_table[state, action] + alpha * \
                             (reward + gamma * np.max(q_table[state2, :]) - q_table[state, action])


def my_get_state(agent_id):
    unit = env.get_unit_by_id(agent_id)
    target_items = env.enemies.items()
    shoot_range = env.unit_shoot_range(agent_id)
    can_attack = False

    for t_id, t_unit in target_items:
        if t_unit.health > 0:
            dist = env.distance(
                unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y
            )
            if dist <= shoot_range:
                can_attack = True

    if not can_attack:
        state = (unit.pos.x - map_size[0]) * env_states_range[1] + (unit.pos.y - map_size[2])
    else:
        state = (unit.pos.x - map_size[0]) * env_states_range[1] + (unit.pos.y - map_size[2]) + \
                n_states // 2 - 1

    return int(state)


method_name = "CQT"
env = StarCraft2Env(map_name="2m_vs_2zg_DQN")
env_info = env.get_env_info()

map_size = (8, 24, 25, 28)  # x1, x2, y1, y2
env_states_range = (map_size[1] - map_size[0], map_size[3] - map_size[2])
n_actions = env_info["n_actions"]
n_agents = env_info["n_agents"]
episode_limit = env_info["episode_limit"]

n_episodes = 2200
n_states = env_states_range[0] * env_states_range[1] * 2

alpha = 0.2
gamma = 0.95
epsilon = 0.7
epsilon_start_decay = n_episodes // 4
epsilon_decay_rate = (1 / (n_episodes - epsilon_start_decay - n_episodes / 20)) * (1 - epsilon)

q_table = np.zeros([n_states, n_actions])

epsilon_values = []
total_steps = []
ep_reward = []
mean_total_steps = []
mean_ep_reward = []
ep_reward_agent1 = []
ep_reward_agent2 = []
mean_ep_reward_agent1 = []
mean_ep_reward_agent2 = []

for e in tqdm(range(1, n_episodes + 1), ascii=True, unit="episode"):
    env.reset()
    terminated = False
    episode_reward = 0
    episode_reward1 = 0
    episode_reward2 = 0
    n_steps = 1

    if e > epsilon_start_decay:
        if epsilon < 1:
            epsilon += epsilon_decay_rate

    while not terminated:
        states = []
        next_states = []
        actions = []
        rewards = []

        for agent_id in range(n_agents):
            state = my_get_state(agent_id)
            states.append(state)
            avail_actions = env.get_avail_agent_actions(agent_id)
            avail_actions[1] = 0
            avail_actions_ind = np.nonzero(avail_actions)[0]
            action = choose_action(state, avail_actions_ind)
            if action > 5:
                rewards.append(10)
            else:
                rewards.append(-2)
            actions.append(action)

        _, terminated, _ = env.step(actions)

        for i in range(len(rewards)):
            rewards[i] = rewards[i] * ((episode_limit - n_steps + 50) / episode_limit) ** 6

        episode_reward += sum(rewards)

        for agent_id in range(n_agents):
            next_state = my_get_state(agent_id)
            next_states.append(next_state)

        for agent_id in range(n_agents):
            learn(states[agent_id], next_states[agent_id], rewards[agent_id], actions[agent_id])

        episode_reward1 += rewards[0]
        episode_reward2 += rewards[1]
        n_steps += 1

    ep_reward.append(episode_reward)
    total_steps.append(n_steps)
    ep_reward_agent1.append(episode_reward1)
    ep_reward_agent2.append(episode_reward2)

    if not e % 10:
        epsilon_values.append(epsilon)
        mean_ep_reward.append(np.mean(ep_reward[-10:]))
        mean_total_steps.append(np.mean(total_steps[-10:]))
        mean_ep_reward_agent1.append(np.mean(ep_reward_agent1[-10:]))
        mean_ep_reward_agent2.append(np.mean(ep_reward_agent2[-10:]))

    # if (not e % 100) and (e != 0):
    #     with open(f'2v2_common_q_table_ep{e}.pkl', 'wb') as f:
    #         pickle.dump(q_table, f)

    game_stats = env.get_stats()
    print()
    print('Episode ', e)
    print('Steps: {}   Reward: {}'.format(n_steps, round(episode_reward, 3)))
    print('Won: {}    Played: {}    Win rate: {}'.format(game_stats['battles_won'],
                                                         game_stats['battles_game'],
                                                         round(game_stats['win_rate'], 3)))

# with open("2v2_common_q_table_final.pkl", 'wb') as f:
#     pickle.dump(q_table, f)

x = np.linspace(0, n_episodes, n_episodes // 10)

with open(f"{method_name}_plot_mean_ep_reward_agent1", 'wb') as f:
    pickle.dump(mean_ep_reward_agent1, f)

with open(f"{method_name}_plot_mean_ep_reward_agent2", 'wb') as f:
    pickle.dump(mean_ep_reward_agent2, f)

with open(f"{method_name}_plot_mean_ep_reward", 'wb') as f:
    pickle.dump(mean_ep_reward, f)

with open(f"{method_name}_plot_mean_total_steps", 'wb') as f:
    pickle.dump(mean_total_steps, f)

with open(f"{method_name}_plot_x", 'wb') as f:
    pickle.dump(x, f)

fig, ax = plt.subplots(1, 3)
ax1, ax2, ax3 = ax.flatten()

ax1.plot(x, epsilon_values)
ax1.set_title('Epsilon')
ax2.plot(x, mean_ep_reward)
ax2.set_title('Rewards')
ax3.plot(x, mean_total_steps)
ax3.set_title('Steps')
fig.set_size_inches(15, 4)
fig.subplots_adjust(hspace=0.2, wspace=0.2)
plt.show()

env.close()
