from smac.env import StarCraft2Env
import numpy as np
import nashpy as nash
import random
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from collections import deque
import time
import os

# map_name = "2m_vs_2zg"
map_name = "2m_vs_2zg_DQN_testing"

MODEL_NAME = "64"

# path to model file (ex: 128x64___.h5) OR None if new model
LOAD_MODEL = "64__449.84avg_x2.h5"
# LOAD_MODEL = None

env = StarCraft2Env(map_name=map_name)
env_info = env.get_env_info()
state_shape = env_info["state_shape"]
obs_shape = env_info["obs_shape"]
n_actions = env_info["n_actions"]
n_agents = env_info["n_agents"]
episode_limit = env_info["episode_limit"]

EPS_WITH_EPSILON = 500
EPS_WITH_ZERO_EPSILON = 100
EPISODES = EPS_WITH_EPSILON + EPS_WITH_ZERO_EPSILON
ALPHA = 0.001
GAMMA = 0.99

epsilon = 0
EPSILON_DECAY_RATE = 1 / EPS_WITH_EPSILON
MIN_EPSILON = 0

# stats for matplotlib
epsilon_values = []
total_steps = []
ep_reward = []
mean_total_steps = []
mean_ep_reward = []
ep_reward_agent1 = []
ep_reward_agent2 = []
mean_ep_reward_agent1 = []
mean_ep_reward_agent2 = []

REPLAY_MEMORY_SIZE = 4000  # n last steps of env for training
MIN_REPLAY_MEMORY_SIZE = 10000  # minimum n of steps in a memory to start training
BATCH_SIZE = 64  # samples for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
AGGREGATE_STATS_EVERY = 20  # episodes

ep_rewards = [0]
new_average_reward = 500

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

if not os.path.isdir('models_nash'):
    os.makedirs('models_nash')


class Experience:
    def __init__(self):
        # queue, that provide append and pop to store replays
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    def update_replay_memory(self, transition):
        # step of env is transaction - (current_state, actions, rewards, new_state, new_action, terminated)
        self.replay_memory.append(transition)

    def get_len(self):
        return len(self.replay_memory)

    def get_batch(self):
        return random.sample(self.replay_memory, BATCH_SIZE)


class DQN:
    def __init__(self, id_of_agent):
        self.agent_id = id_of_agent

        # evaluation network model (trains every step)
        self.model = self.create_model()

        # target network model (predicts every step, updates on UPDATE_TARGET_EVERY)
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.target_update_counter = 0

    def create_model(self):
        if LOAD_MODEL is not None:
            model = load_model(f"models_nash/ag_id{self.agent_id}_{LOAD_MODEL}")
            print(f"Loaded model: {LOAD_MODEL}")
        else:
            model = Sequential()
            model.add(Dense(64, input_shape=(state_shape,)))
            model.add(Activation("relu"))
            model.add(Dropout(0.2))

            # model.add(Dense(64, input_shape=(64,)))
            # model.add(Activation("relu"))
            # model.add(Dropout(0.2))

            model.add(Dense(n_actions, activation="linear"))
            model.compile(loss="mse", optimizer=Adam(lr=ALPHA), metrics=["accuracy"])
        return model

    def get_q_values(self, state):
        # from eval net
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

    def next_predictions(self, batch):
        # from target net
        new_current_states = np.array([transition[3] for transition in batch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)

        return future_qs_list

    def train(self, batch, nash_q_values, terminal_state):
        # from eval net
        current_states = np.array([transition[0] for transition in batch]) / 255
        current_qs_list = self.model.predict(current_states)

        # observations and actions (a.k.a. features and labels)
        X = []
        y = []

        for index, (current_state, actions, rewards, new_state, new_actions, terminated) in enumerate(batch):
            if not terminated:
                new_q = rewards[self.agent_id] + GAMMA * nash_q_values[index][self.agent_id]
            else:
                new_q = rewards[self.agent_id]

            # update q
            current_qs = current_qs_list[index]
            current_qs[actions[self.agent_id]] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X) / 255, np.array(y), batch_size=BATCH_SIZE,
                       verbose=0, shuffle=False, callbacks=None)

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            # updating weights of target net
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


def nash_equilibrium(actions, qs):
    agent_1_act_num = len(actions[0])
    agent_1_values = np.zeros(agent_1_act_num)
    agent_1_keys = np.arange(agent_1_act_num)
    agent_1_decoder = dict(zip(agent_1_keys, actions[0]))

    for action_index in range(agent_1_act_num):
        agent_1_values[action_index] = qs[0][agent_1_decoder[action_index]]

    agent_2_act_num = len(actions[1])
    agent_2_values = np.zeros(agent_2_act_num)
    agent_2_keys = np.arange(agent_2_act_num)
    agent_2_decoder = dict(zip(agent_2_keys, actions[1]))

    for action_index in range(agent_2_act_num):
        agent_2_values[action_index] = qs[1][agent_2_decoder[action_index]]

    matrix_size = [len(agent_1_values), len(agent_2_values)]
    reward_matrix = np.zeros(matrix_size)

    for row in range(matrix_size[0]):
        for column in range(matrix_size[1]):
            reward_matrix[row][column] = agent_1_values[row] + agent_2_values[column]

    reward_matrix = reward_matrix * 10
    shape = reward_matrix.shape
    if shape[0] > shape[1]:
        # more rows than columns, add columns
        new_columns = np.random.sample((shape[0], shape[0] - shape[1])) / 10
        reward_matrix = np.hstack((reward_matrix, new_columns))
    elif shape[0] < shape[1]:
        # more columns than rows, add rows
        new_rows = np.random.sample((shape[1] - shape[0], shape[1])) / 10
        reward_matrix = np.vstack((reward_matrix, new_rows))

    game = nash.Game(reward_matrix, reward_matrix.T)

    for idl in range(sum(reward_matrix.shape) - 1):
        try:
            nash_eq = game.lemke_howson(initial_dropped_label=idl)
            break
        except:
            continue

    try:
        agent_1_nash_q = agent_1_values[nash_eq[1].argmax()]
    except:
        agent_1_nash_q = agent_1_values.max()

    try:
        agent_2_nash_q = agent_2_values[nash_eq[0].argmax()]
    except:
        agent_2_nash_q = agent_2_values.max()

    return [agent_1_nash_q, agent_2_nash_q]


# list with DQN classes for each agent
agents = [DQN(i) for i in range(n_agents)]

# set id for every agent
# for agent_id, agent in enumerate(agents):
#     agent.agent_id = agent_id

replay_memory = Experience()

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episode"):
    env.reset()
    terminated = False
    episode_reward = 0
    episode_reward1 = 0
    episode_reward2 = 0
    n_steps = 1

    if epsilon > MIN_EPSILON:
        epsilon -= EPSILON_DECAY_RATE
    else:
        epsilon = MIN_EPSILON

    while not terminated:
        actions = []
        rewards = []
        state = env.get_state()

        for agent_id in range(n_agents):
            try:
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions[1] = 0
            except:
                avail_actions = [0] * n_actions
                avail_actions[1] = 1

            avail_actions_ind = np.nonzero(avail_actions)[0]

            if np.random.random() > epsilon:
                q_values = agents[agent_id].get_q_values(state)
                avail_qs = np.array([-100] * len(q_values))

                # get q values of available actions
                for possible_action in avail_actions_ind:
                    avail_qs[possible_action] = q_values[possible_action]

                action = np.argmax(avail_qs)
            else:
                action = np.random.choice(avail_actions_ind)

            actions.append(action)

        # reward assigning
        for agent_id in range(n_agents):
            action = actions[agent_id]
            if action > 5:
                rewards.append(10)
            else:
                rewards.append(-2)

        _, terminated, _ = env.step(actions)

        if sum(rewards) > 0:
            for i in range(len(rewards)):
                rewards[i] = rewards[i] * ((episode_limit - n_steps + 50) / episode_limit) ** 6

        episode_reward += sum(rewards)

        new_avail_actions = []

        for agent_id in range(n_agents):
            try:
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions[1] = 0
            except:
                avail_actions = [0] * n_actions
                avail_actions[1] = 1

            avail_actions_ind = np.nonzero(avail_actions)[0]

            new_avail_actions.append(avail_actions_ind)

        # add step of env to replays and train
        replay_memory.update_replay_memory((state, actions, rewards, env.get_state(), new_avail_actions, terminated))

        if replay_memory.get_len() > MIN_REPLAY_MEMORY_SIZE:
            batch = replay_memory.get_batch()
            future_qs_raw = [agent.next_predictions(batch) for agent in agents]
            future_qs = list(zip(future_qs_raw[0], future_qs_raw[1]))

            nash_qs = []

            for i in range(len(batch)):
                nash_qs.append(nash_equilibrium(batch[i][4], future_qs[i]))

            for agent in agents:
                agent.train(batch, nash_qs, terminated)

        episode_reward1 += rewards[0]
        episode_reward2 += rewards[1]
        n_steps += 1

    ep_rewards.append(episode_reward)

    if not episode % AGGREGATE_STATS_EVERY:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])

        # save model if it improved
        if average_reward >= new_average_reward:
            for agent in agents:
                agent.model.save(
                    f"models_nash/ag_id{agent.agent_id}_{MODEL_NAME}_{average_reward:_>7.2f}avg_{max_reward:_>7.2f}"
                    f"max_{min_reward:_>7.2f}min__{int(time.time())}.h5")

            new_average_reward = average_reward

    # for matplotlib
    ep_reward.append(episode_reward)
    total_steps.append(n_steps)
    ep_reward_agent1.append(episode_reward1)
    ep_reward_agent2.append(episode_reward2)

    if not episode % 10:
        epsilon_values.append(epsilon)
        mean_ep_reward.append(np.mean(ep_reward[-10:]))
        mean_total_steps.append(np.mean(total_steps[-10:]))
        mean_ep_reward_agent1.append(np.mean(ep_reward_agent1[-10:]))
        mean_ep_reward_agent2.append(np.mean(ep_reward_agent2[-10:]))

    game_stats = env.get_stats()
    print()
    print('Episode ', episode)
    print(f"Steps: {n_steps}   Reward: {round(episode_reward, 3)}   Epsilon: {round(epsilon, 3)}")
    print('Won: {}    Played: {}    Win rate: {}'.format(game_stats['battles_won'],
                                                         game_stats['battles_game'],
                                                         round(game_stats['win_rate'], 3)))

x = np.linspace(0, EPISODES, EPISODES // 10)

with open(f"{map_name}_{MODEL_NAME}_DQN_plot_mean_ep_reward_agent1", 'wb') as f:
    pickle.dump(mean_ep_reward_agent1, f)

with open(f"{map_name}_{MODEL_NAME}_DQN_plot_mean_ep_reward_agent2", 'wb') as f:
    pickle.dump(mean_ep_reward_agent2, f)

with open(f"{map_name}_{MODEL_NAME}_DQN_plot_epsilon_values", 'wb') as f:
    pickle.dump(epsilon_values, f)

with open(f"{map_name}_{MODEL_NAME}_DQN_plot_mean_ep_reward", 'wb') as f:
    pickle.dump(mean_ep_reward, f)

with open(f"{map_name}_{MODEL_NAME}_DQN_plot_mean_total_steps", 'wb') as f:
    pickle.dump(mean_total_steps, f)

with open(f"{map_name}_{MODEL_NAME}_DQN_plot_x", 'wb') as f:
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
