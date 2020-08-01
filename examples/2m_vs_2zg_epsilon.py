import numpy as np
import matplotlib.pyplot as plt

n_episodes = 10000
EPSILON = 0.6
FICT_PROB = 0.1
epsilon_values = []
fict_values = []

fict_prob = FICT_PROB

for e in range(n_episodes):

    n_steps = 1
    # fict_prob += (1 - fict_prob) ** 0.5 * 0.9 / 1000
    fict_prob += (1 - fict_prob) ** 0.1 / n_episodes

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

    epsilon_values.append(epsilon)
    fict_values.append(fict_prob)

x = np.linspace(0, n_episodes, n_episodes)

fig, ax = plt.subplots(1, 2)
ax1, ax2 = ax.flatten()

ax1.plot(x, epsilon_values)
ax1.set_title('Epsilon')

ax2.plot(x, fict_values)
ax2.set_title('Fictious play probability')

fig.set_size_inches(15, 4)
fig.subplots_adjust(hspace=0.2, wspace=0.2)
plt.show()
