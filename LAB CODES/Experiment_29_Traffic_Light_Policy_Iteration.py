import numpy as np

# States represent traffic density
states = 4

# Actions
# 0 = Short Green
# 1 = Long Green
actions = 2

gamma = 0.9

# Transition probabilities
# [state, action, next_state]
P = np.zeros((states, actions, states))

for s in range(states):

    if s == 0:
        P[s, 0, 0] = 1.0
        P[s, 1, 0] = 1.0

    elif s == 1:
        P[s, 0, 1] = 0.7
        P[s, 0, 0] = 0.3

        P[s, 1, 0] = 0.7
        P[s, 1, 1] = 0.3

    elif s == 2:
        P[s, 0, 2] = 0.8
        P[s, 0, 1] = 0.2

        P[s, 1, 1] = 0.8
        P[s, 1, 0] = 0.2

    else:
        P[s, 0, 3] = 0.9
        P[s, 0, 2] = 0.1

        P[s, 1, 2] = 0.7
        P[s, 1, 1] = 0.3


# Negative reward = waiting vehicles
R = np.array([-1, -5, -10, -20])

policy = np.zeros(states, dtype=int)

for iteration in range(20):

    # Policy Evaluation
    V = np.zeros(states)

    for _ in range(100):

        new_V = np.zeros(states)

        for s in range(states):

            a = policy[s]

            new_V[s] = R[s] + gamma * np.sum(
                P[s, a] * V
            )

        V = new_V

    # Policy Improvement
    stable = True

    for s in range(states):

        old_action = policy[s]

        action_values = []

        for a in range(actions):

            value = R[s] + gamma * np.sum(
                P[s, a] * V
            )

            action_values.append(value)

        policy[s] = np.argmax(action_values)

        if old_action != policy[s]:
            stable = False

    if stable:
        break


print("=" * 65)
print("POLICY ITERATION FOR TRAFFIC LIGHT CONTROL")
print("=" * 65)

print("\nState Meaning:")
print("0 = Very Low Traffic")
print("1 = Low Traffic")
print("2 = Medium Traffic")
print("3 = High Traffic")

print("\nOptimal Policy:")

for s in range(states):

    action = "Short Green" if policy[s] == 0 else "Long Green"

    print("State", s, "->", action)

print("\nOptimal State Values:")
print(np.round(V, 2))

print("\nResult: Optimal traffic-light policy obtained.")