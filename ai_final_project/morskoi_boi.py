import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

grid = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
])
size = 5
Q = np.zeros((size, size, 4))
actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # left, right, up, down

alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
episodes = 1000

def is_valid(pos):
    i, j = pos
    return 0 <= i < size and 0 <= j < size

def step(pos, action_idx):
    move = actions[action_idx]
    new_pos = (pos[0] + move[0], pos[1] + move[1])
    if not is_valid(new_pos):
        return pos, -5, False
    if grid[new_pos] == 1:
        return new_pos, 10, True
    return new_pos, -1, False

# обучение
rewards = []
selected_episodes = [0, 61, 125, episodes - 1]
episode_paths = {}

for episode in range(episodes):
    pos = (0, 0)
    done = False
    total_reward = 0
    path = [pos]

    while not done:
        if random.random() < epsilon:
            action_idx = random.randint(0, 3)
        else:
            action_idx = np.argmax(Q[pos[0], pos[1]])

        new_pos, reward, done = step(pos, action_idx)

        i, j = pos
        ni, nj = new_pos
        Q[i, j, action_idx] += alpha * (reward + gamma * np.max(Q[ni, nj]) - Q[i, j, action_idx])

        pos = new_pos
        path.append(pos)
        total_reward += reward

    if episode in selected_episodes:
        episode_paths[episode] = path

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    rewards.append(total_reward)
    print(f"Эпизод {episode + 1}: Награда = {total_reward}")

# Анимация
fig, ax = plt.subplots(figsize=(5, 5))

def draw_grid():
    ax.clear()
    for i in range(size):
        for j in range(size):
            color = 'black' if grid[i, j] == 1 else 'white'
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color, ec='gray'))
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_xticks([])
    ax.set_yticks([])

all_frames = []
for ep, path in episode_paths.items():
    for step in path:
        all_frames.append((ep, step))

def update(frame_data):
    ep, pos = frame_data
    draw_grid()
    i, j = pos
    ax.plot(j + 0.5, i + 0.5, 'ro', markersize=14)
    ax.set_title(f"Анимация: эпизод {ep + 1}")
    return []

ani = animation.FuncAnimation(fig, update, frames=all_frames, interval=500, repeat=False)

# Сохраняем видео
writer = FFMpegWriter(fps=2, metadata=dict(artist='Me'), bitrate=1800)
ani.save("animation.mp4", writer=writer)
print("Анимация сохранена как 'animation.mp4'")


plt.figure(figsize=(6, 4))
plt.plot(rewards)
plt.xlabel("Эпизод")
plt.ylabel("Суммарная награда")
plt.title("Награды по эпизодам")
plt.grid(True)
plt.savefig("rewards_plot.png")

plt.show()
