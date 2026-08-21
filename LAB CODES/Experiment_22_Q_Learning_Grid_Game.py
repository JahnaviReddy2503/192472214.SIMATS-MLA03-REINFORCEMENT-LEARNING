import numpy as np
import random

SIZE = 5
states = SIZE * SIZE
actions = 4

# 0 = Up, 1 = Down, 2 = Left, 3 = Right
action_names = ["Up", "Down", "Left", "Right"]

# Food and ghost positions
food = 24
ghost = 12

Q = np.zeros((states, actions))

alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

episodes = 1000

def move(state, action):

    row = state // SIZE
    col = state % SIZE

    if action == 0:
        row = max(0, row - 1)
    elif action == 1:
        row = min(SIZE - 1, row + 1)
    elif action == 2:
        col = max(0, col - 1)
    else:
        col = min(SIZE - 1, col + 1)

    return row * SIZE + col


for episode in range(episodes):

    state = 0

    for step in range(100):

        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[state])

        next_state = move(state, action)

        if next_state == food:
            reward = 100
            done = True
        elif next_state == ghost:
            reward = -100
            done = True
        else:
            reward = -1
            done = False

        Q[state, action] += alpha * (
            reward +
            gamma * np.max(Q[next_state]) * (not done)
            - Q[state, action]
        )

        state = next_state

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)


print("=" * 55)
print("Q-LEARNING GRID GAME")
print("=" * 55)

print("\nTraining completed.")
print("Food State:", food)
print("Ghost State:", ghost)

state = 0
path = [state]

for step in range(30):

    action = np.argmax(Q[state])
    state = move(state, action)
    path.append(state)

    if state == food:
        print("\nFood collected!")
        break

    if state == ghost:
        print("\nAgent reached the ghost.")
        break

print("Learned Path:", path)
print("Number of Moves:", len(path) - 1)