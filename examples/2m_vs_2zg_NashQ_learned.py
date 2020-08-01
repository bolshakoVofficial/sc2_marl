from smac.env import StarCraft2Env
import numpy as np
import pickle


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


def choose_action(agent_id, state, avail_actions_ind):
    qt_arr = np.zeros(len(avail_actions_ind))
    keys = np.arange(len(avail_actions_ind))
    act_ind_decode = dict(zip(keys, avail_actions_ind))

    for act_ind in range(len(avail_actions_ind)):
        qt_arr[act_ind] = q_table[agent_id, state, act_ind_decode[act_ind]]

    action = act_ind_decode[np.argmax(qt_arr)]

    return action


with open("2v2_NashQ_ep10000.pkl", 'rb') as f:
    q_table = pickle.load(f)

map_name = "2m_vs_2zg_DQN_testing"
# map_name = "2m_vs_2zg"

env = StarCraft2Env(map_name=map_name)
env_info = env.get_env_info()
n_agents = env_info["n_agents"]
n_episodes = 100

if map_name == "2m_vs_2zg_DQN_testing":
    map_size = (7, 27, 25, 27)  # x1, x2, y1, y2
elif map_name == "2m_vs_2zg":
    map_size = (8, 24, 25, 28)  # x1, x2, y1, y2
else:
    map_size = (1, 1, 1, 1)

env_states_range = (map_size[1] - map_size[0], map_size[3] - map_size[2])
n_states = env_states_range[0] * env_states_range[1] * 2

for e in range(n_episodes):
    env.reset()
    terminated = False
    actions_history1 = []
    actions_history2 = []

    while not terminated:
        actions = []
        for agent_id in range(n_agents):
            state = my_get_state(agent_id)
            avail_actions = env.get_avail_agent_actions(agent_id)
            avail_actions[1] = 0
            avail_actions_ind = np.nonzero(avail_actions)[0]
            action = choose_action(agent_id, state, avail_actions_ind)
            actions.append(action)

        reward, terminated, _ = env.step(actions)
        actions_history1.append(actions[0])
        actions_history2.append(actions[1])

    print(actions_history1)
    print(actions_history2)

env.close()
