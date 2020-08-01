from smac.env import StarCraft2Env
import numpy as np
import matplotlib.pyplot as plt


def generateRandomFromDistribution(distribution):
    randomIndex = 0
    randomSum = distribution[randomIndex]
    randomFlag = np.random.random_sample()
    while randomFlag > randomSum:
        randomIndex += 1
        randomSum += distribution[randomIndex]
    return randomIndex


class agent:
    def __init__(self, initialStrategy=(0.5, 0.5), gammma=0.9, delta=0.0001):
        self.timeStep = 0
        self.alpha = 1 / (10 + 0.00001 * self.timeStep)
        self.gamma = gammma
        self.actions = [0, 1]
        self.lengthOfAction = len(self.actions)
        self.reward = 0.0
        self.strategy = list(initialStrategy)
        self.actionValues = np.zeros((self.lengthOfAction))  # Q(s, a)
        self.currentAction = np.random.choice(self.actions)
        self.currentReward = 0
        self.maxAction = np.random.choice(self.actions)
        self.EPSILON = 0.5 / (1 + 0.0001 * self.timeStep)
        self.deltaAction = np.zeros((self.lengthOfAction))
        self.deltaActionTop = np.zeros((self.lengthOfAction))
        self.delta = delta
        # WoLF parameters
        self.stateCount = 0.0
        self.averageStrategy = np.zeros((self.lengthOfAction))
        for i in self.actions:
            self.averageStrategy[i] = 1.0 / self.lengthOfAction
        self.deltaWin = 0.0
        self.deltaLose = 0.0

    def initialSelfStrategy(self):
        for i in range(self.lengthOfAction):
            self.strategy[i] = 1.0 / self.lengthOfAction

    def initialActionValues(self):
        for i in range(self.lengthOfAction):
            self.actionValues[i] = 0

    def chooseAction(self):
        if np.random.binomial(1, self.EPSILON) == 1:
            self.currentAction = np.random.choice(self.actions)
        else:
            self.currentAction = self.actions[generateRandomFromDistribution(self.strategy)]

    def chooseActionWithFxiedStrategy(self):
        self.currentAction = self.actions[generateRandomFromDistribution(self.strategy)]

    def getCurrentAction(self):
        return self.currentAction

    def setReward(self, agentReward):
        self.currentReward = agentReward

    def updateActionValues(self):
        self.actionValues[self.currentAction] = (1 - self.alpha) * self.actionValues[self.currentAction] \
                                                + self.alpha * (self.currentReward + self.gamma * np.amax(
            self.actionValues[:]))

    def updateStrategy(self):
        self.stateCount += 1.0
        self.deltaWin = 20.0 / (20000 + self.timeStep)
        self.deltaLose = 2.0 * self.deltaWin
        for action_i in self.actions:
            self.averageStrategy[action_i] += (1.0 / self.stateCount) * (
                    self.strategy[action_i] - self.averageStrategy[action_i])
        self.sumActionValue = 0.0
        self.sumAverageActionValue = 0.0
        for action_i in self.actions:
            self.sumActionValue += self.strategy[action_i] * self.actionValues[action_i]
            self.sumAverageActionValue += self.averageStrategy[action_i] * self.actionValues[action_i]
        if self.sumActionValue > self.sumAverageActionValue:
            self.delta = self.deltaWin
        else:
            self.delta = self.deltaLose

        self.maxAction = np.argmax(self.actionValues)
        for i in range(self.lengthOfAction):
            self.deltaAction[i] = np.min([self.strategy[i], self.delta / (self.lengthOfAction - 1)])
        self.sumDeltaAction = 0
        for action_i in [action_j for action_j in self.actions if action_j != self.maxAction]:
            self.deltaActionTop[action_i] = -self.deltaAction[action_i]
            self.sumDeltaAction += self.deltaAction[action_i]
        self.deltaActionTop[self.maxAction] = self.sumDeltaAction
        for i in range(self.lengthOfAction):
            self.strategy[i] += self.deltaActionTop[i]

        # if self.currentAction != self.maxAction:
        #     self.deltaActionTop[self.currentAction] = -self.deltaAction[self.currentAction]
        # else:
        #     self.sumDeltaAction = 0
        #     for action_i in [action_j for action_j in self.actions if action_j != self.currentAction]:
        #         self.sumDeltaAction += self.deltaAction[action_i]
        #     self.deltaActionTop[self.currentAction] = self.sumDeltaAction
        # self.strategy[self.currentAction] += self.deltaActionTop[self.currentAction]

    def updateTimeStep(self):
        self.timeStep += 1

    def updateEpsilon(self):
        self.EPSILON = 0.5 / (1 + 0.0001 * self.timeStep)

    def updateAlpha(self):
        self.alpha = 1 / (10 + 0.00001 * self.timeStep)


env = StarCraft2Env(map_name="2m_vs_2zg_WoLF_testing")
env_info = env.get_env_info()

n_actions = env_info["n_actions"]
n_agents = env_info["n_agents"]
n_episodes = 700
overall_episode_reward = 0
overall_steps = 0

forward_both = np.array([[4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1,
                          4, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 4, 1, 4, 1, 4, 1, 4, 1,
                          4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 1, 1, 1, 1]])

hide_both = np.array([[5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1,
                       5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1],
                      [5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1,
                       5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1]])

forward_hide = np.array([[4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1,
                          4, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1,
                          5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1]])

hide_forward = np.array([[5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1,
                          5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 4, 1, 4, 1, 4, 1, 4, 1,
                          4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 1, 1, 1, 1]])

a = agent(initialStrategy=(0.5, 0.5))
b = agent(initialStrategy=(0.5, 0.5))
action_to_display = 1
aStrategyActionZero = []
aStrategyActionZero.append(a.strategy[action_to_display])
bStrategyActionZero = []
bStrategyActionZero.append(b.strategy[action_to_display])

for e in range(n_episodes):
    env.reset()
    terminated = False
    episode_reward = 0
    n_steps = 1

    if not e % 10:
        print('Episode:', e)
        print('Agent0: forward = {}, hide = {}'.format(a.strategy[1], a.strategy[0]))
        print('Agent1: forward = {}, hide = {}'.format(b.strategy[1], b.strategy[0]))
        print()

    a.chooseAction()
    actionA = a.getCurrentAction()
    b.chooseAction()
    actionB = b.getCurrentAction()

    if (actionA, actionB) == (1, 1):
        strategy = forward_both
    elif (actionA, actionB) == (0, 0):
        strategy = hide_both
    elif (actionA, actionB) == (1, 0):
        strategy = forward_hide
    elif (actionA, actionB) == (0, 1):
        strategy = hide_forward

    while not terminated:
        obs = env.get_obs()
        actions = []
        for agent_id in range(n_agents):
            if n_steps < len(strategy[0]):
                action = strategy[agent_id][n_steps - 1]
            else:
                avail_actions = env.get_avail_agent_actions(agent_id)
                if avail_actions[6] == 1:
                    action = 6
                elif avail_actions[7] == 1:
                    action = 7
                elif avail_actions[1] == 1:
                    action = 1
                else:
                    avail_actions_ind = np.nonzero(avail_actions)[0]
                    action = np.random.choice(avail_actions_ind)
            actions.append(action)

        reward, terminated, _ = env.step(actions)
        reward = reward / n_steps
        n_steps += 1
        episode_reward += reward

    rewardA, rewardB = episode_reward, episode_reward
    a.setReward(rewardA)
    b.setReward(rewardB)
    a.updateActionValues()
    b.updateActionValues()
    a.updateStrategy()
    b.updateStrategy()
    a.updateTimeStep()
    b.updateTimeStep()
    a.updateEpsilon()
    b.updateEpsilon()
    a.updateAlpha()
    b.updateAlpha()
    aStrategyActionZero.append(a.strategy[action_to_display])
    bStrategyActionZero.append(b.strategy[action_to_display])

    '''print("n_steps = {} Total reward in episode {} = {}".format(n_steps, e, episode_reward))
    print(env.get_stats())
    overall_episode_reward += episode_reward
    overall_steps += n_steps'''

plt.figure(1)
plt.plot(aStrategyActionZero, label='Agent0')
plt.plot(bStrategyActionZero, label='Agent1')
plt.legend()
plt.xlabel('timestep')
plt.ylabel('probability')
plt.show()

env.close()
