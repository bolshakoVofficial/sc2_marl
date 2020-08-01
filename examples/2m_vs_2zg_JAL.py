from smac.env import StarCraft2Env
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm


def choose_action(states, possible_ja):
    if random.uniform(0, 1) < (1 - epsilon):
        action = np.random.choice(possible_ja)
    else:
        qt_arr1 = np.zeros(len(possible_ja))
        qt_arr2 = np.zeros(len(possible_ja))

        keys = np.arange(len(possible_ja))
        act_ind_decode = dict(zip(keys, possible_ja))

        for act_ind in range(len(possible_ja)):
            qt_arr1[act_ind] = q_table[states[0], act_ind_decode[act_ind]]
            qt_arr2[act_ind] = q_table[states[1], act_ind_decode[act_ind]]

        action1 = act_ind_decode[np.argmax(qt_arr1)]
        q_value1 = max(qt_arr1)

        action2 = act_ind_decode[np.argmax(qt_arr2)]
        q_value2 = max(qt_arr2)

        if q_value1 > q_value2:
            action = action1
        else:
            action = action2

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

    if state > q_table.shape[0] - 1:
        print('Agent position when out of bounds with state', unit.pos.x, unit.pos.y, state)
        state = q_table.shape[0] - 1

    return int(state)


def jal_encoder(action1, action2):
    return action1 * n_actions + action2


def jal_decoder(action):
    return [action // n_actions, action % n_actions]


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


method_name = "JAL"
env = StarCraft2Env(map_name="2m_vs_2zg")
env_info = env.get_env_info()

map_size = (8, 24, 25, 28)  # x1, x2, y1, y2
env_states_range = (map_size[1] - map_size[0], map_size[3] - map_size[2])
n_actions = env_info["n_actions"]
n_agents = env_info["n_agents"]
episode_limit = env_info["episode_limit"]

n_episodes = 10_200
n_states = env_states_range[0] * env_states_range[1] * 2

alpha = 0.2
gamma = 0.95
EPSILON = 0.6
q_table = np.zeros([n_states, n_actions * n_actions])

epsilon_values = []
total_steps = []
ep_reward = []
mean_total_steps = []
mean_ep_reward = []
ep_reward_agent1 = []
ep_reward_agent2 = []
mean_ep_reward_agent1 = []
mean_ep_reward_agent2 = []

epsilon = EPSILON

for e in tqdm(range(1, n_episodes + 1), ascii=True, unit="episode"):
    env.reset()
    terminated = False
    episode_reward = 0
    episode_reward1 = 0
    episode_reward2 = 0
    n_steps = 1

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
        all_avail_actions = []
        next_states = []
        actions = []
        rewards = []

        for agent_id in range(n_agents):
            state = my_get_state(agent_id)
            states.append(state)

            avail_actions = env.get_avail_agent_actions(agent_id)
            avail_actions[1] = 0
            avail_actions_ind = np.nonzero(avail_actions)[0]
            all_avail_actions.append(avail_actions_ind)

        possible_joint_actions = avail_joint_actions(all_avail_actions)

        joint_action = choose_action(states, possible_joint_actions)

        actions_pair = jal_decoder(joint_action)

        for agent_id in range(n_agents):
            action = actions_pair[agent_id]
            if action > 5:
                rewards.append(10)
            else:
                rewards.append(-2)
            actions.append(action)

        _, terminated, _ = env.step(actions)

        for i in range(len(rewards)):
            rewards[i] = rewards[i] * ((episode_limit - n_steps + 50) / episode_limit) ** 6

        for agent_id in range(n_agents):
            next_state = my_get_state(agent_id)
            next_states.append(next_state)

        for agent_id in range(n_agents):
            learn(states[agent_id], next_states[agent_id], sum(rewards), joint_action)

            if (states[0] == states[1]) and (next_states[0] == next_states[1]):
                break

        episode_reward1 += rewards[0]
        episode_reward2 += rewards[1]
        episode_reward += sum(rewards)
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
        with open(f'2v2_jal_ep{e}.pkl', 'wb') as f:
            pickle.dump(q_table, f)

    game_stats = env.get_stats()
    print()
    print('Episode ', e)
    print('Steps: {}   Reward: {}'.format(n_steps, round(episode_reward, 3)))
    print('Won: {}    Played: {}    Win rate: {}'.format(game_stats['battles_won'],
                                                         game_stats['battles_game'],
                                                         round(game_stats['win_rate'], 3)))

with open("2v2_jal_final.pkl", 'wb') as f:
    pickle.dump(q_table, f)

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
