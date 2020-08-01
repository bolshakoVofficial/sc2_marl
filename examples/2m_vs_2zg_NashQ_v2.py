# ищем Nash Eq по матрице наград из Q-занчений

from smac.env import StarCraft2Env
import numpy as np
import nashpy as nash
import random
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm


def choose_action(agent_id, state, avail_actions_ind):
    if random.uniform(0, 1) < (1 - epsilon):
        action = np.random.choice(avail_actions_ind)  # Explore action space
    else:
        qt_arr = np.zeros(len(avail_actions_ind))
        keys = np.arange(len(avail_actions_ind))
        act_ind_decode = dict(zip(keys, avail_actions_ind))

        for act_ind in range(len(avail_actions_ind)):
            qt_arr[act_ind] = q_table[agent_id, state, act_ind_decode[act_ind]]

        action = act_ind_decode[np.argmax(qt_arr)]  # Exploit learned values
    return action


def nash_equilibrium(states, actions):
    agent_1_act_num = len(actions[0])
    agent_1_values = np.zeros(agent_1_act_num)
    agent_1_keys = np.arange(agent_1_act_num)
    agent_1_decoder = dict(zip(agent_1_keys, actions[0]))

    for action_index in range(agent_1_act_num):
        agent_1_values[action_index] = q_table[0, states[0], agent_1_decoder[action_index]]

    agent_2_act_num = len(actions[1])
    agent_2_values = np.zeros(agent_2_act_num)
    agent_2_keys = np.arange(agent_2_act_num)
    agent_2_decoder = dict(zip(agent_2_keys, actions[1]))

    for action_index in range(agent_2_act_num):
        agent_2_values[action_index] = q_table[1, states[1], agent_2_decoder[action_index]]

    matrix_size = [len(agent_1_values), len(agent_2_values)]
    reward_matrix = np.zeros(matrix_size)

    for row in range(matrix_size[0]):
        for column in range(matrix_size[1]):
            reward_matrix[row][column] = agent_1_values[row] + agent_2_values[column]

    reward_matrix = reward_matrix * 10
    shape = reward_matrix.shape
    if shape[0] > shape[1]:
        # more rows than columns, add columns
        # new_columns = np.random.randint(-10, 0, [shape[0], shape[0] - shape[1]])
        # new_columns = np.zeros([shape[0], shape[0] - shape[1]])
        new_columns = np.random.sample((shape[0], shape[0] - shape[1])) / 10
        reward_matrix = np.hstack((reward_matrix, new_columns))
    elif shape[0] < shape[1]:
        # more columns than rows, add rows
        # new_rows = np.random.randint(-10, 0, [shape[1] - shape[0], shape[1]])
        # new_rows = np.zeros([shape[1] - shape[0], shape[1]])
        new_rows = np.random.sample((shape[1] - shape[0], shape[1])) / 10
        reward_matrix = np.vstack((reward_matrix, new_rows))

    game = nash.Game(reward_matrix, reward_matrix.T)
    nash_eq = game.lemke_howson(initial_dropped_label=0)

    try:
        agent_1_nashQ = agent_1_values[nash_eq[1].argmax()]
    except:
        agent_1_nashQ = agent_1_values.max()

    try:
        agent_2_nashQ = agent_2_values[nash_eq[0].argmax()]
    except:
        agent_2_nashQ = agent_2_values.max()

    # print(f"Agent 1\nChosen: {agent_1_nashQ}, \nBest: {agent_1_values.max()}\n")
    # print(f"Agent 2\nChosen: {agent_2_nashQ}, \nBest: {agent_2_values.max()}\n")

    return [agent_1_nashQ, agent_2_nashQ]


def learn(agent_id, state, nash_q, reward, action):
    q_table[agent_id, state, action] = q_table[agent_id, state, action] + alpha * \
                                       (reward + gamma * nash_q - q_table[agent_id, state, action])


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


map_name = "2m_vs_2zg_DQN"
# map_name = "2m_vs_2zg"
env = StarCraft2Env(map_name=map_name)
env_info = env.get_env_info()

if map_name == "2m_vs_2zg_DQN":
    map_size = (7, 27, 25, 27)  # x1, x2, y1, y2
elif map_name == "2m_vs_2zg":
    map_size = (8, 24, 25, 28)  # x1, x2, y1, y2
else:
    map_size = (1, 1, 1, 1)

env_states_range = (map_size[1] - map_size[0], map_size[3] - map_size[2])

n_actions = env_info["n_actions"]
n_agents = env_info["n_agents"]
episode_limit = env_info["episode_limit"]

n_episodes = 10_000
n_states = env_states_range[0] * env_states_range[1] * 2

alpha = 0.2
gamma = 0.95
EPSILON = 0.6
epsilon = EPSILON
epsilon_start_decay = n_episodes // 4
epsilon_decay_rate = (1 / (n_episodes - epsilon_start_decay - n_episodes / 20)) * (1 - epsilon)

q_table = np.random.rand(n_agents, n_states, n_actions)

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

    # if e > epsilon_start_decay:
    #     if epsilon < 1:
    #         epsilon += epsilon_decay_rate

    if e < 500:
        epsilon = EPSILON
    elif e < 2000:
        epsilon += ((1 - epsilon) ** 0.5) * 0.8 / 1000
    elif e < 3000:
        epsilon -= ((1 - epsilon) ** 0.5) * 1.03 / 1000
    elif e < 3500:
        epsilon = EPSILON + 0.1
    elif e < 5000:
        epsilon += ((1 - epsilon) ** 0.5) * 0.7 / 1000
    elif e < 6000:
        epsilon -= ((1 - epsilon) ** 0.5) * 0.85 / 1000
    elif e < 6500:
        epsilon = EPSILON + 0.2
    elif e < 8000:
        epsilon += ((1 - epsilon) ** 0.5) * 0.58 / 1000
    elif e < 8500:
        epsilon -= ((1 - epsilon) ** 0.5) * 1.2 / 1000
    elif e < 9000:
        epsilon = EPSILON + 0.3
    elif e < 10000:
        epsilon += ((1 - epsilon) ** 0.5) * 0.6 / 1000
    else:
        epsilon = 1

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
            action = choose_action(agent_id, state, avail_actions_ind)
            if (action == 6) or (action == 7):
                rewards.append(10)
            else:
                rewards.append(0)
            actions.append(action)

        _, terminated, _ = env.step(actions)

        for i in range(len(rewards)):
            rewards[i] = rewards[i] * ((episode_limit - n_steps + 50) / episode_limit) ** 6

        episode_reward += sum(rewards)

        for agent_id in range(n_agents):
            next_state = my_get_state(agent_id)
            next_states.append(next_state)

        new_avail_acts = []
        all_new_acts = []
        for agent_id in range(n_agents):
            avail_actions = env.get_avail_agent_actions(agent_id)
            all_new_acts.append(avail_actions)
            avail_actions[1] = 0
            avail_actions_ind = np.nonzero(avail_actions)[0]
            new_avail_acts.append(avail_actions_ind)

        if all_new_acts[0][0] == 1 or all_new_acts[1][0] == 1:
            if all_new_acts[0][0] == 1:
                nash_q_values = [0, max(q_table[1, next_states[1], :])]
            else:
                nash_q_values = [max(q_table[0, next_states[0], :]), 0]
        else:
            nash_q_values = nash_equilibrium(next_states, new_avail_acts)

        for agent_id in range(n_agents):
            learn(agent_id, states[agent_id], nash_q_values[agent_id],
                  rewards[agent_id], actions[agent_id])

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

    if (not e % 100) and (e != 0):
        with open(f'2v2_NashQ_ep{e}.pkl', 'wb') as f:
            pickle.dump(q_table, f)

    game_stats = env.get_stats()
    print()
    print('Episode ', e)
    print('Steps: {}   Reward: {}'.format(n_steps, round(episode_reward, 3)))
    print('Won: {}    Played: {}    Win rate: {}'.format(game_stats['battles_won'],
                                                         game_stats['battles_game'],
                                                         round(game_stats['win_rate'], 3)))

with open(f"2v2_NashQ_final_{map_name}.pkl", 'wb') as f:
    pickle.dump(q_table, f)

with open(f"2v2_NashQ_plot_mean_ep_reward_agent2_{map_name}", 'wb') as f:
    pickle.dump(mean_ep_reward_agent1, f)

with open(f"2v2_NashQ_plot_mean_ep_reward_agent1_{map_name}", 'wb') as f:
    pickle.dump(mean_ep_reward_agent2, f)

with open(f"2v2_NashQ_plot_epsilon_values_{map_name}", 'wb') as f:
    pickle.dump(epsilon_values, f)

with open(f"2v2_NashQ_plot_mean_ep_reward_{map_name}", 'wb') as f:
    pickle.dump(mean_ep_reward, f)

with open(f"2v2_NashQ_plot_mean_total_steps_{map_name}", 'wb') as f:
    pickle.dump(mean_total_steps, f)

x = np.linspace(0, n_episodes, n_episodes // 10)

with open(f"2v2_NashQ_plot_x_{map_name}", 'wb') as f:
    pickle.dump(x, f)

plt.plot(x, mean_ep_reward_agent1, label='Agent0 reward')
plt.plot(x, mean_ep_reward_agent2, label='Agent1 reward')
plt.legend()

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
