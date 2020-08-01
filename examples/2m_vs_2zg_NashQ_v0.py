# Nash Eq рассчитываем на матрице наград из расстояний до цели

from smac.env import StarCraft2Env
import numpy as np
import nashpy as nash
import random
import pickle
import matplotlib.pyplot as plt


def choose_action(states, avail_actions_list):
    actions = []
    rewards = []

    if random.uniform(0, 1) < (1 - epsilon):
        for agent in range(n_agents):
            avail_actions_ind = np.nonzero(avail_actions_list[agent])[0]
            actions.append(np.random.choice(avail_actions_ind))

    # if one agent is dead or some agent can attack
    elif avail_actions_list[0][0] == 1 or avail_actions_list[1][0] == 1 or \
            avail_actions_list[0][6] == 1 or avail_actions_list[0][7] == 1 or \
            avail_actions_list[1][6] == 1 or avail_actions_list[1][7] == 1:

        for agent in range(n_agents):
            avail_actions_ind = np.nonzero(avail_actions_list[agent])[0]
            state = states[agent]

            qt_arr = np.zeros(len(avail_actions_ind))
            keys = np.arange(len(avail_actions_ind))
            act_ind_decode = dict(zip(keys, avail_actions_ind))

            for act_ind in range(len(avail_actions_ind)):
                qt_arr[act_ind] = q_table[state, act_ind_decode[act_ind]]

            actions.append(act_ind_decode[np.argmax(qt_arr)])
    else:
        reward_matrix = fill_reward_matrix(avail_actions_list)
        nash_game = nash.Game(reward_matrix, reward_matrix.T)
        eq = nash_game.lemke_howson(initial_dropped_label=0)
        agent_1_action = eq[0].argmax() + 2
        agent_2_action = eq[1].argmax() + 2
        actions.extend([agent_1_action, agent_2_action])
        common_reward = reward_matrix[eq[0].argmax()][eq[1].argmax()]

    try:
        if actions[0] > 5:
            # attack
            reward = max(map_size[1] - map_size[0], map_size[3] - map_size[2]) * 2
            rewards.append(reward)
        elif actions[0] < 2:
            # dead
            rewards.append(0)
        elif (2 <= actions[0] <= 5) and actions[1] > 1:
            # moving
            try:
                rewards.append(common_reward)
            except:
                reward_matrix = fill_reward_matrix(avail_actions_list)
                nash_game = nash.Game(reward_matrix, reward_matrix.T)
                eq = nash_game.lemke_howson(initial_dropped_label=0)
                common_reward = reward_matrix[eq[0].argmax()][eq[1].argmax()]

                rewards.append(common_reward)
        else:
            rewards.append(0)
    except:
        rewards.append(0)

    try:
        if actions[1] > 5:
            # attack
            reward = max(map_size[1] - map_size[0], map_size[3] - map_size[2]) * 2
            rewards.append(reward)
        elif actions[1] < 2:
            # dead
            rewards.append(0)
        elif (2 <= actions[1] <= 5) and actions[0] > 1:
            # moving
            try:
                rewards.append(common_reward)
            except:
                reward_matrix = fill_reward_matrix(avail_actions_list)
                nash_game = nash.Game(reward_matrix, reward_matrix.T)
                eq = nash_game.lemke_howson(initial_dropped_label=0)
                common_reward = reward_matrix[eq[0].argmax()][eq[1].argmax()]

                rewards.append(common_reward)
        else:
            rewards.append(0)
    except:
        rewards.append(0)

    return actions, rewards


def fill_reward_matrix(avail_actions_list):
    agent_1_moves = avail_actions_list[0][2:6]
    agent_2_moves = avail_actions_list[1][2:6]
    reward_matrix = np.zeros([4, 4])
    agent_1_rewards = measure_distance(0, agent_1_moves)
    agent_2_rewards = measure_distance(1, agent_2_moves)

    for row in range(len(agent_1_moves)):
        for column in range(len(agent_2_moves)):
            reward_matrix[row][column] = agent_1_rewards[row] + agent_2_rewards[column]

    return reward_matrix


def measure_distance(agent, directions):
    unit = env.get_unit_by_id(agent)
    target_items = env.enemies.items()
    min_distance = max(map_size[1] - map_size[0], map_size[3] - map_size[2]) + 10

    for t_id, t_unit in target_items:
        if t_unit.health > 0:
            if min_distance > env.distance(unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y):
                min_distance = env.distance(unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y)
                enemy = t_unit

    simulated_distances = []
    max_distance = max(map_size[1] - map_size[0], map_size[3] - map_size[2]) + 1

    # up
    if directions[0] == 1:
        north = env.distance(unit.pos.x, unit.pos.y - 0.25, enemy.pos.x, enemy.pos.y)
        simulated_distances.append(max_distance - north)
    else:
        simulated_distances.append(0)

    # down
    if directions[1] == 1:
        south = env.distance(unit.pos.x, unit.pos.y + 0.25, enemy.pos.x, enemy.pos.y)
        simulated_distances.append(max_distance - south)
    else:
        simulated_distances.append(0)

    # right
    if directions[2] == 1:
        east = env.distance(unit.pos.x + 0.25, unit.pos.y, enemy.pos.x, enemy.pos.y)
        simulated_distances.append(max_distance - east)
    else:
        simulated_distances.append(0)

    # left
    if directions[3] == 1:
        west = env.distance(unit.pos.x - 0.25, unit.pos.y, enemy.pos.x, enemy.pos.y)
        simulated_distances.append(max_distance - west)
    else:
        simulated_distances.append(0)

    return simulated_distances


def learn(state, state2, reward, action):
    q_table[state, action] = q_table[state, action] + alpha * (reward - q_table[state, action])


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


env = StarCraft2Env(map_name="2m_vs_2zg_NashQ")
env_info = env.get_env_info()

map_size = (7, 33, 25, 27)  # x1, x2, y1, y2
env_states_range = (map_size[1] - map_size[0], map_size[3] - map_size[2])

n_actions = env_info["n_actions"]
n_agents = env_info["n_agents"]
episode_limit = env_info["episode_limit"]

n_episodes = 400
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

for e in range(n_episodes):
    env.reset()
    terminated = False
    episode_reward = 0
    n_steps = 1

    if e > epsilon_start_decay:
        if epsilon < 1:
            epsilon += epsilon_decay_rate

    while not terminated:
        states = []
        next_states = []
        available_actions = []
        rewards = []
        moves_vector = []

        for agent_id in range(n_agents):
            state = my_get_state(agent_id)
            states.append(state)
            avail_actions = env.get_avail_agent_actions(agent_id)
            avail_actions[1] = 0
            available_actions.append(avail_actions)

        actions, rewards = choose_action(states, available_actions)
        _, terminated, _ = env.step(actions)

        for i in range(len(rewards)):
            rewards[i] = rewards[i] * ((episode_limit - n_steps + 50) / episode_limit) ** 6

        episode_reward += sum(rewards)

        for agent_id in range(n_agents):
            next_state = my_get_state(agent_id)
            next_states.append(next_state)

        for agent_id in range(n_agents):
            learn(states[agent_id], next_states[agent_id], rewards[agent_id], actions[agent_id])

        n_steps += 1

    ep_reward.append(episode_reward)
    total_steps.append(n_steps)

    if not e % 10:
        epsilon_values.append(epsilon)
        mean_ep_reward.append(np.mean(ep_reward[-10:]))
        mean_total_steps.append(np.mean(total_steps[-10:]))

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

with open("2v2_common_q_table_final.pkl", 'wb') as f:
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
