import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# ==========================================================
# PATIENT HEALTH MONITORING AND TREATMENT RECOMMENDATION
# MDP + BELLMAN EQUATION + BANDIT
# ==========================================================

# -----------------------------
# 1. MDP: STATES AND ACTIONS
# -----------------------------

states = ["Stable", "Moderate Risk", "Severe Risk"]

actions = [
    "No Intervention",
    "Low Treatment",
    "Moderate Treatment",
    "Urgent Intervention"
]

# -----------------------------
# 2. REWARD FUNCTION
# -----------------------------

rewards = {
    "Stable": {
        "No Intervention": 5,
        "Low Treatment": 3,
        "Moderate Treatment": 1,
        "Urgent Intervention": -3
    },

    "Moderate Risk": {
        "No Intervention": -5,
        "Low Treatment": 6,
        "Moderate Treatment": 8,
        "Urgent Intervention": 5
    },

    "Severe Risk": {
        "No Intervention": -10,
        "Low Treatment": -3,
        "Moderate Treatment": 5,
        "Urgent Intervention": 10
    }
}

# -----------------------------
# 3. TRANSITION PROBABILITIES
# -----------------------------

transitions = {

    "Stable": {
        "No Intervention": [0.90, 0.09, 0.01],
        "Low Treatment": [0.95, 0.05, 0.00],
        "Moderate Treatment": [0.97, 0.03, 0.00],
        "Urgent Intervention": [0.98, 0.02, 0.00]
    },

    "Moderate Risk": {
        "No Intervention": [0.10, 0.60, 0.30],
        "Low Treatment": [0.40, 0.50, 0.10],
        "Moderate Treatment": [0.70, 0.25, 0.05],
        "Urgent Intervention": [0.80, 0.18, 0.02]
    },

    "Severe Risk": {
        "No Intervention": [0.02, 0.18, 0.80],
        "Low Treatment": [0.05, 0.35, 0.60],
        "Moderate Treatment": [0.25, 0.55, 0.20],
        "Urgent Intervention": [0.70, 0.25, 0.05]
    }
}

# ==========================================================
# 4. BELLMAN VALUE ITERATION
# ==========================================================

gamma = 0.90
V = np.zeros(len(states))

for iteration in range(100):

    new_V = np.zeros(len(states))

    for i, state in enumerate(states):

        values = []

        for action in actions:

            reward = rewards[state][action]

            probability = transitions[state][action]

            future_value = np.sum(
                np.array(probability) * V
            )

            value = reward + gamma * future_value

            values.append(value)

        new_V[i] = max(values)

    if np.max(np.abs(new_V - V)) < 0.0001:
        V = new_V
        break

    V = new_V

# -----------------------------
# 5. OPTIMAL POLICY
# -----------------------------

optimal_policy = {}

for i, state in enumerate(states):

    values = []

    for action in actions:

        reward = rewards[state][action]

        probability = transitions[state][action]

        future_value = np.sum(
            np.array(probability) * V
        )

        value = reward + gamma * future_value

        values.append(value)

    optimal_policy[state] = actions[np.argmax(values)]

print("\n========== BELLMAN RESULTS ==========")

for state, value in zip(states, V):
    print(f"{state}: V* = {value:.2f}")

print("\n========== OPTIMAL POLICY ==========")

for state, action in optimal_policy.items():
    print(f"{state} -> {action}")

# ==========================================================
# 6. MULTI-ARMED BANDIT
# ==========================================================

true_values = np.array([2.0, 4.5, 6.5, 7.5])

n_actions = len(actions)
episodes = 1000


def bandit(epsilon):

    Q = np.zeros(n_actions)
    N = np.zeros(n_actions)

    rewards_history = []
    actions_history = []

    for t in range(episodes):

        # Exploration / Exploitation
        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q)

        reward = np.random.normal(
            true_values[action],
            2
        )

        N[action] += 1

        Q[action] += (
            reward - Q[action]
        ) / N[action]

        rewards_history.append(reward)
        actions_history.append(action)

    return (
        np.array(rewards_history),
        np.array(actions_history),
        Q
    )


# ==========================================================
# 7. COMPARE EXPLORATION STRATEGIES
# ==========================================================

epsilons = [0.0, 0.1, 0.2, 0.5]

results = {}

for epsilon in epsilons:

    reward_history, action_history, Q = bandit(epsilon)

    results[epsilon] = {
        "rewards": reward_history,
        "actions": action_history,
        "Q": Q
    }

# ==========================================================
# 8. DISPLAY VALUE ESTIMATES
# ==========================================================

print("\n========== BANDIT VALUE ESTIMATES ==========")

for epsilon in epsilons:

    print(f"\nEpsilon = {epsilon}")

    for action, value in zip(
        actions,
        results[epsilon]["Q"]
    ):
        print(f"{action}: {value:.2f}")

# ==========================================================
# 9. CUMULATIVE REWARD GRAPH
# ==========================================================

plt.figure(figsize=(10, 6))

for epsilon in epsilons:

    cumulative = np.cumsum(
        results[epsilon]["rewards"]
    )

    plt.plot(
        cumulative,
        label=f"Epsilon = {epsilon}"
    )

plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward Comparison")
plt.legend()
plt.grid()
plt.show()

# ==========================================================
# 10. MOVING AVERAGE REWARD
# ==========================================================

plt.figure(figsize=(10, 6))

window = 50

for epsilon in epsilons:

    reward = results[epsilon]["rewards"]

    moving_average = np.convolve(
        reward,
        np.ones(window) / window,
        mode="valid"
    )

    plt.plot(
        moving_average,
        label=f"Epsilon = {epsilon}"
    )

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Learning Behaviour")
plt.legend()
plt.grid()
plt.show()

# ==========================================================
# 11. ACTION SELECTION
# ==========================================================

for epsilon in epsilons:

    counts = np.bincount(
        results[epsilon]["actions"],
        minlength=n_actions
    )

    plt.figure(figsize=(8, 5))

    plt.bar(actions, counts)

    plt.xlabel("Treatment Action")
    plt.ylabel("Number of Selections")

    plt.title(
        f"Action Selection - Epsilon = {epsilon}"
    )

    plt.xticks(rotation=20)

    plt.grid(axis="y")

    plt.show()

# ==========================================================
# 12. PERFORMANCE COMPARISON
# ==========================================================

performance = []

for epsilon in epsilons:

    reward = results[epsilon]["rewards"]

    performance.append({

        "Strategy": f"Epsilon-Greedy ({epsilon})",

        "Average Reward":
            np.mean(reward),

        "Cumulative Reward":
            np.sum(reward),

        "Best Action":
            actions[
                np.argmax(
                    results[epsilon]["Q"]
                )
            ]
    })

performance_df = pd.DataFrame(performance)

print("\n========== PERFORMANCE COMPARISON ==========")
print(performance_df.to_string(index=False))

# ==========================================================
# 13. BEST BANDIT STRATEGY
# ==========================================================

best_index = performance_df[
    "Average Reward"
].idxmax()

print("\n========== BEST STRATEGY ==========")

print(
    performance_df.loc[
        best_index
    ].to_string()
)

# ==========================================================
# 14. FINAL SUMMARY
# ==========================================================

print("\n==========================================")
print("FINAL PATIENT HEALTH RL SYSTEM")
print("==========================================")

print("\nStates:")
print(states)

print("\nActions:")
print(actions)

print("\nOptimal Bellman Policy:")

for state, action in optimal_policy.items():
    print(f"{state} -> {action}")

print("\nBest Bandit Strategy:")
print(
    performance_df.loc[
        best_index,
        "Strategy"
    ]
)

print("\nBest Bandit Action:")
print(
    performance_df.loc[
        best_index,
        "Best Action"
    ]
)

print("\nSystem successfully completed.")