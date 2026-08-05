import numpy as np
import random

# Autonomous Drone Delivery Environment

states = 10
actions = 4

# Actions
action_names = [
    "Move Forward",
    "Move Left",
    "Move Right",
    "Return to Base"
]

# Q Network (Q Table representation)
Q = np.zeros((states, actions))

# Parameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0
epsilon_decay = 0.9
min_epsilon = 0.1

episodes = 20


print("="*60)
print("DQN Based Autonomous Drone Delivery System")
print("="*60)


# Training

for episode in range(episodes):

    state = random.randint(0, states-1)

    total_reward = 0

    for step in range(10):

        # Exploration and Exploitation

        if random.random() < epsilon:
            action = random.randint(0, actions-1)

        else:
            action = np.argmax(Q[state])


        # Simulated drone environment

        next_state = random.randint(0, states-1)


        # Reward design

        if action == 3:
            reward = 20       # Returning safely

        elif next_state == 9:
            reward = 50       # Delivery completed

        else:
            reward = -1       # Battery usage


        # Q value update

        old_value = Q[state][action]

        best_future = np.max(Q[next_state])

        new_value = old_value + learning_rate * (
            reward + discount_factor * best_future - old_value
        )


        Q[state][action] = new_value


        total_reward += reward

        state = next_state


    epsilon = max(min_epsilon, epsilon * epsilon_decay)


    print("\nEpisode :", episode+1)
    print("Total Reward :", total_reward)
    print("Exploration Rate :", round(epsilon,3))


print("\n")
print("="*60)
print("Training Completed")
print("="*60)


print("\nLearned Q Values")

print(np.round(Q,2))


# Testing

test_state = random.randint(0, states-1)

best_action = np.argmax(Q[test_state])


print("\nTesting Drone")

print("Current Drone State :", test_state)

print("Recommended Action :", 
      action_names[best_action])


print("\nDrone Delivery Optimization Completed")