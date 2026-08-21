import numpy as np
import random

# Actions:
# 0 = Sell
# 1 = Hold
# 2 = Buy

actions = ["Sell", "Hold", "Buy"]

state_size = 3
action_size = 3

theta = np.zeros((state_size, action_size))

learning_rate = 0.01
episodes = 100


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def policy(state):
    return softmax(np.dot(state, theta))


for episode in range(episodes):

    states = []
    chosen_actions = []
    rewards = []

    for step in range(20):

        state = np.random.randn(state_size)

        probabilities = policy(state)

        action = np.random.choice(
            action_size,
            p=probabilities
        )

        market_return = random.uniform(-1, 1)

        if action == 2:
            reward = market_return
        elif action == 0:
            reward = -market_return
        else:
            reward = 0.1

        states.append(state)
        chosen_actions.append(action)
        rewards.append(reward)

    # Calculate returns
    returns = []
    G = 0

    for reward in reversed(rewards):
        G = reward + 0.95 * G
        returns.insert(0, G)

    # REINFORCE update
    for state, action, G in zip(
        states, chosen_actions, returns
    ):

        probabilities = policy(state)

        gradient = -probabilities
        gradient[action] += 1

        theta += learning_rate * G * np.outer(
            state,
            gradient
        )


print("=" * 55)
print("REINFORCE AUTOMATED TRADING")
print("=" * 55)

test_state = np.array([0.7, -0.2, 0.5])

probabilities = policy(test_state)

print("\nTest Market State:", test_state)
print("Action Probabilities:",
      np.round(probabilities, 3))

best_action = np.argmax(probabilities)

print("Recommended Trading Action:",
      actions[best_action])

print("\nTraining completed successfully.")