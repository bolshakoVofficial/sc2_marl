from smac.env import StarCraft2Env
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
import time
import os

map_name = "2m_vs_2zg"
MODEL_NAME = "NATA_128x64_10k"

# path to model file (ex: models/128x64___.h5) OR None
# LOAD_MODEL = "models/"
LOAD_MODEL = None

env = StarCraft2Env(map_name=map_name)
env_info = env.get_env_info()

map_size = (8, 24, 25, 28)  # x1, x2, y1, y2

# for feature map
MULTIPLIER = 4
Y_MULTIPLIER = 4

# (y, x)
game_area = ((map_size[3] - map_size[2] + 1) * MULTIPLIER * Y_MULTIPLIER,
             (map_size[1] - map_size[0]) * MULTIPLIER, 3)

n_actions = env_info["n_actions"]
n_agents = env_info["n_agents"]
episode_limit = env_info["episode_limit"]

EPS_WITH_EPSILON = 9500
EPS_WITH_ZERO_EPSILON = 500
EPISODES = EPS_WITH_EPSILON + EPS_WITH_ZERO_EPSILON
ALPHA = 0.001
GAMMA = 0.95

epsilon = 1
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
MIN_REPLAY_MEMORY_SIZE = 1000  # minimum n of steps in a memory to start training
BATCH_SIZE = 64  # samples for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
AGGREGATE_STATS_EVERY = 20  # episodes

ep_rewards = [0]
new_average_reward = 350
SHOW_FEATURE_MAP = False

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

if not os.path.isdir('models'):
    os.makedirs('models')


class DQN:
    def __init__(self):
        # evaluation network model (trains every step)
        self.model = self.create_model()

        # target network model (predicts every step, updates on UPDATE_TARGET_EVERY)
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # queue, that provide append and pop to store replays
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0

    def create_model(self):
        if LOAD_MODEL is not None:
            model = load_model(LOAD_MODEL)
            print(f"Loaded model: {LOAD_MODEL}")
        else:
            # game_area : (64, 64, 3)
            model = Sequential()
            model.add(Conv2D(128, (8, 8), strides=(4, 4), input_shape=game_area))
            model.add(Activation("relu"))
            model.add(Dropout(0.2))

            model.add(Conv2D(64, (4, 4), strides=(2, 2)))
            model.add(Activation("relu"))
            model.add(Dropout(0.2))

            model.add(Flatten())
            model.add(Dense(256))
            model.add(Dropout(0.2))

            model.add(Dense(n_actions ** 2, activation="linear"))
            model.compile(loss="mse", optimizer=Adam(lr=ALPHA), metrics=["accuracy"])
        return model

    def update_replay_memory(self, transition):
        # step of env is transition - (current_state, action, reward, new_state, terminated)
        self.replay_memory.append(transition)

    def get_q_values(self, state):
        # from eval net
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            # not enough replays for learning
            return

        # train on random replays from memory
        batch = random.sample(self.replay_memory, BATCH_SIZE)

        # from eval net
        current_states = np.array([transition[0] for transition in batch]) / 255
        current_qs_list = self.model.predict(current_states)

        # from target net
        new_current_states = np.array([transition[3] for transition in batch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)

        # observations and actions (a.k.a. features and labels)
        X = []
        y = []

        for index, (current_state, action, reward, new_state, terminated) in enumerate(batch):
            if not terminated:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + GAMMA * max_future_q
            else:
                new_q = reward

            # update q
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

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


def draw_feature_map():
    f_map = np.zeros((game_area[0], game_area[1], 3), np.uint8)

    for agent in range(n_agents):
        ally_unit = env.get_unit_by_id(agent)
        if ally_unit.health > 0:
            cv2.circle(f_map, (int((map_size[1] - (round(ally_unit.pos.x, 1))) * MULTIPLIER),
                               int((map_size[3] - round(ally_unit.pos.y, 1)) * MULTIPLIER * Y_MULTIPLIER)), 1, (0, 255, 0), -1)

    enemies = env.enemies.items()
    for enemy_id, enemy_unit in enemies:
        if enemy_unit.health > 0:
            cv2.circle(f_map, (int((map_size[1] - (round(enemy_unit.pos.x, 1))) * MULTIPLIER),
                               int((map_size[3] - round(enemy_unit.pos.y, 1)) * MULTIPLIER * Y_MULTIPLIER)), 1, (0, 0, 255), -1)

    return f_map


network = DQN()

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
        all_avail_actions = []
        actions = []
        rewards = []

        # this "image" will be used by CNN
        feature_map = draw_feature_map()

        for agent_id in range(n_agents):
            avail_actions = env.get_avail_agent_actions(agent_id)
            avail_actions[1] = 0
            avail_actions_ind = np.nonzero(avail_actions)[0]
            all_avail_actions.append(avail_actions_ind)

        # joint actions that satisfy restrictions of env
        possible_joint_actions = avail_joint_actions(all_avail_actions)

        if SHOW_FEATURE_MAP:
            flipped_image = cv2.flip(feature_map, 1)
            resized_image = cv2.resize(flipped_image, dsize=None, fx=1, fy=1)  # 10 times bigger than CNN gets
            cv2.imshow(f"Feature map ({MULTIPLIER} times bigger)", resized_image)
            cv2.waitKey(1)

        if np.random.random() > epsilon:
            q_values = network.get_q_values(feature_map)
            avail_qs = np.array([-100] * len(q_values))

            # get q values of available actions
            for possible_action in possible_joint_actions:
                avail_qs[possible_action] = q_values[possible_action]

            joint_action = np.argmax(avail_qs)
        else:
            joint_action = np.random.choice(possible_joint_actions)

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

        if sum(rewards) > 0:
            for i in range(len(rewards)):
                rewards[i] = rewards[i] * ((episode_limit - n_steps + 50) / episode_limit) ** 6

        episode_reward += sum(rewards)
        new_feature_map = draw_feature_map()

        # add step of env to replays and train
        network.update_replay_memory((feature_map, joint_action, sum(rewards), new_feature_map, terminated))
        network.train(terminated)

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
            network.model.save(
                f"models/{MODEL_NAME}__{average_reward:_>7.2f}avg_{max_reward:_>7.2f}max_{min_reward:_>7.2f}min__{int(time.time())}.h5")

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
