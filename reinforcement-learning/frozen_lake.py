import pygame
import numpy as np
import gymnasium as gym
import random
import seaborn as sns
import matplotlib.pyplot as plt


# ------------------------------ Hyperparameters ----------------------------- #

map_size = 4
map_name = "4x4"
alpha = 0.1
gamma = 0.95
epsilon = 0.1
temperature = 0.1
n_episodes = 20000
is_slippery = True
strategy = "epsilon-greedy"  # "epsilon-greedy" or "boltzmann"

# ------------------------------- Training loop ------------------------------ #

env = gym.make("FrozenLake-v1", map_name=map_name, desc=None, is_slippery=is_slippery)
q_table = np.zeros([env.observation_space.n, env.action_space.n])


for i in range(n_episodes):
    # Initial random state
    state, _ = env.reset()
    terminated, truncated = False, False

    while not (terminated or truncated):

        # Choose a random action with probability epsilon or best action with probability 1 - epsilon
        if strategy == "epsilon-greedy":
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

        # Choose an action based on the Boltzmann distribution
        elif strategy == "boltzmann":
            exp_q = np.exp(q_table[state] / temperature)
            probabilities = exp_q / np.sum(exp_q)
            action = np.random.choice(np.arange(len(probabilities)), p=probabilities)

        # Transition to the next state
        next_state, reward, terminated, truncated, info = env.step(action)
        next_action = np.argmax(q_table[next_state])

        # Update Q-table
        q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (
            reward + gamma * (q_table[next_state, next_action])
        )

        state = next_state


# --------------------------------- Plotting --------------------------------- #


def qtable_directions_map(qtable, map_size):
    """Get the best learned action & map it to arrows."""
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            # Assign an arrow only if a minimal Q-value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where
            # it didn't actually learn anything
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions


def plot_q_values_map(qtable, env, map_size):
    """Plot the last frame of the simulation and the policy learned."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # Plot the policy
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    plt.show()


def plot(q_table, map_size, is_slippery=True):
    env = gym.make("FrozenLake-v1", render_mode="rgb_array", map_name=f"{map_size}x{map_size}", is_slippery=is_slippery)
    env.reset()
    plot_q_values_map(q_table, env, map_size)


plot(q_table, map_size, is_slippery)

# --------------------------------- Gameplay --------------------------------- #


def play(q_table, map_size, is_slippery=True):
    env = gym.make("FrozenLake-v1", render_mode="human", map_name=f"{map_size}x{map_size}", is_slippery=is_slippery)

    while True:
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
