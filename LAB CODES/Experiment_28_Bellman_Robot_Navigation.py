import numpy as np

SIZE = 4
gamma = 0.9

# Goal state
goal = (3, 3)

# Value function
V = np.zeros((SIZE, SIZE))

actions = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1)
]

action_names = ["Up", "Down", "Left", "Right"]


def next_state(state, action):

    r, c = state

    dr, dc = actions[action]

    nr = min(max(r + dr, 0), SIZE - 1)
    nc = min(max(c + dc, 0), SIZE - 1)

    return nr, nc


for iteration in range(100):

    new_V = V.copy()

    for r in range(SIZE):
        for c in range(SIZE):

            if (r, c) == goal:
                continue

            values = []

            for action in range(4):

                ns = next_state((r, c), action)

                reward = 10 if ns == goal else -1

                value = reward + gamma * V[ns]

                values.append(value)

            new_V[r, c] = max(values)

    if np.max(np.abs(new_V - V)) < 0.001:
        V = new_V
        break

    V = new_V


# Find optimal path
state = (0, 0)
path = [state]

for _ in range(20):

    if state == goal:
        break

    values = []

    for action in range(4):

        ns = next_state(state, action)

        reward = 10 if ns == goal else -1

        values.append(reward + gamma * V[ns])

    action = np.argmax(values)

    state = next_state(state, action)

    path.append(state)


print("=" * 60)
print("BELLMAN OPTIMALITY ROBOT NAVIGATION")
print("=" * 60)

print("\nOptimal State Values:")
print(np.round(V, 2))

print("\nOptimal Path:")

for state in path:
    print(state, end=" ")

print("\n\nGoal State:", goal)
print("Result: Optimal navigation policy obtained.")