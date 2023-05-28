import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import random
import pygame
import numpy as np

# ---------------------------------------------------------------------------- #
#                                  Environment                                 #
# ---------------------------------------------------------------------------- #

# OBSERVATION SPACE - All possible states

# Passenger locations:
# 0: Red
# 1: Green
# 2: Yellow
# 3: Blue
# 4: In taxi

# Destinations:
# 0: Red
# 1: Green
# 2: Yellow
# 3: Blue

# Taxi location:
# One of the 5x5 grid positions

# Observation space has 5 x 4 x 5x5 = 500 distinct states
# Each state is represented by a number ((passenger_location + 1) x (destionation + 1) x (taxi_location + 1))

# ---------------------------------------------------------------------------- #

# ACTION SPACE - All possible actions for each state

# 0: Move south (down)
# 1: Move north (up)
# 2: Move east (right)
# 3: Move west (left)
# 4: Pickup passenger
# 5: Drop off passenger

# ---------------------------------------------------------------------------- #

# REWARDS

# -1 per step unless other reward is triggered.
# -10 executing “pickup” and “drop-off” actions illegally (penalty).
# +20 delivering passenger (reward).

# ---------------------------------------------------------------------------- #

env = gym.make('Taxi-v3')

print("State space:", env.observation_space)  # All states
print("Action space:", env.action_space)      # All actions for each state

q_table = np.zeros([env.observation_space.n, env.action_space.n])
"""Q-table holds Q-values for each state and each action in a matrix. 
It is filled during the learning process and used to make decisions after the algorithm is trained."""


# ---------------------------------------------------------------------------- #
#                                Hyperparameters                               #
# ---------------------------------------------------------------------------- #


alpha = 0.1
"""Alpha is the learning rate (0 < α ≤ 1) —
Just like in supervised learning settings,
α is the extent to which our Q-values are
being updated in every iteration."""

gamma = 0.6
"""Gamma is the discount factor (0 ≤ γ ≤ 1) —
determines how much importance we want to
give to future rewards. A high value for
the discount factor (close to 1) captures
the long-term effective award, whereas, a
discount factor of 0 makes our agent consider
only immediate reward, hence making it greedy."""

epsilon = 0.1
"""Epsilon is the probability of exploring a
random action instead of selecting the best
learned Q-value. Higher epsilon values result
in episodes with more penalties."""

n_episodes = 100000
"""Number of training episodes - each episode
lasts from the initial state until reaching
terminating state - success or fail - or
truncating state - for example timeout"""


# ---------------------------------------------------------------------------- #
#                                 Training loop                                #
# ---------------------------------------------------------------------------- #

env = TimeLimit(env, max_episode_steps=200)
terminated, truncated = False, False
""""The episode ends if the following happens:
Termination: The taxi drops off the passenger.
Truncation: The length of the episode is 200."""

all_epochs = []
all_penalties = []

for i in range(n_episodes + 1):

    state, _ = env.reset()
    epochs, penalties, reward, = 0, 0, 0
    terminated, truncated = False, False

    while not (terminated or truncated):

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, info = env.step(action)
        next_action = np.argmax(q_table[next_state])

        q_table[state, action] = (1 - alpha) * q_table[state, action] + \
            alpha * (reward + gamma * (q_table[next_state, next_action]))

        if reward == -10:
            penalties += 1
        epochs += 1
        state = next_state

    all_epochs.append(epochs)
    all_penalties.append(penalties)

    if i % 100 == 0:
        print(f"Episode: {i}")

print("Training finished.\n")
print(f"Results after {n_episodes} episodes:")
print(f"Average timesteps per episode: {sum(all_epochs) / n_episodes}")
print(f"Average penalties per episode: {sum(all_penalties) / n_episodes}")


while True:
    env = gym.make('Taxi-v3', render_mode="human")
    env = TimeLimit(env, 50)
    state, _ = env.reset()
    terminated, truncated = False, False

    while not (terminated or truncated):

        try:
            env.render()
            best_action = np.argmax(q_table[state])
            state, _, terminated, truncated, _ = env.step(best_action)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    env.close()

        except KeyboardInterrupt:
            pygame.quit()
            env.close()

    print("Succeeded!") if terminated else print("Did not succed... :(")
