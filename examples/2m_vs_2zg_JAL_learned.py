from smac.env import StarCraft2Env
import numpy as np
import pickle


def choose_action(states, possible_ja):
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


with open("2v2_jal_new_10k.pkl", 'rb') as f:
    q_table = pickle.load(f)

env = StarCraft2Env(map_name="2m_vs_2zg_testing")
env_info = env.get_env_info()

n_actions = env_info["n_actions"]
n_agents = env_info["n_agents"]

n_episodes = 100
map_size = (8, 24, 25, 28)  # x1, x2, y1, y2
env_states_range = (map_size[1] - map_size[0], map_size[3] - map_size[2])
n_states = env_states_range[0] * env_states_range[1] * 2

for e in range(n_episodes):
    env.reset()
    terminated = False
    agent0 = []
    agent1 = []

    while not terminated:
        states = []
        all_avail_actions = []

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

        _, terminated, _ = env.step(actions_pair)

        agent0.append(actions_pair[0])
        agent1.append(actions_pair[1])

    print(agent0)
    print(agent1)

env.close()
