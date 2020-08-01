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


env = StarCraft2Env(map_name="1m_vs_1zg")
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

for e in tqdm(range(1, n_episodes + 1), ascii=True, unit="episode"):
    env.reset()
    terminated = False
    episode_reward = 0
    reward = 0
    n_steps = 1

    if e > epsilon_start_decay:
        if epsilon < 1:
            epsilon += epsilon_decay_rate

    while not terminated:
        actions = []

        for agent_id in range(n_agents):
            state = my_get_state(agent_id)
            avail_actions = env.get_avail_agent_actions(agent_id)
            avail_actions[1] = 0
            avail_actions_ind = np.nonzero(avail_actions)[0]
            action = choose_action(state, avail_actions_ind)
            if action == 6:
                reward = 10
            else:
                reward = -2
            actions.append(action)

        _, terminated, _ = env.step(actions)

        reward = reward * ((episode_limit - n_steps + 50) / episode_limit) ** 6
        episode_reward += reward

        for agent_id in range(n_agents):
            next_state = my_get_state(agent_id)

        learn(state, next_state, reward, action)

        n_steps += 1
        reward = 0

    ep_reward.append(episode_reward)
    total_steps.append(n_steps)

    if not e % 10:
        epsilon_values.append(epsilon)
        mean_ep_reward.append(np.mean(ep_reward[-10:]))
        mean_total_steps.append(np.mean(total_steps[-10:]))

    if (not e % 100) and (e != 0):
        with open(f'1v1_ep{e}.pkl', 'wb') as f:
            pickle.dump(q_table, f)

    game_stats = env.get_stats()
    print()
    print('Episode ', e)
    print('Steps: {}   Reward: {}'.format(n_steps, round(episode_reward, 3)))
    print('Won: {}    Played: {}    Win rate: {}'.format(game_stats['battles_won'],
                                                         game_stats['battles_game'],
                                                         round(game_stats['win_rate'], 3)))

with open("1v1_final.pkl", 'wb') as f:
    pickle.dump(q_table, f)

x = np.linspace(0, n_episodes, n_episodes // 10)
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
