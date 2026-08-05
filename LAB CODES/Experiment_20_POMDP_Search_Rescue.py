import numpy as np
import random

# States
states = ["Safe Zone", "Obstacle", "Victim Found", "Unknown Area"]

# Actions
actions = ["Move Forward", "Turn Left", "Turn Right", "Rescue"]

# Belief State (Initial Probability)
belief = np.array([0.25, 0.25, 0.25, 0.25])

alpha = 0.2
episodes = 10

print("=" * 75)
print("Partially Observable Markov Decision Process (POMDP)")
print("=" * 75)

for episode in range(episodes):

    print("\nEpisode", episode + 1)

    # Observation received from sensors
    observation = random.randint(0, len(states)-1)

    # Update belief state
    belief[observation] += alpha

    belief = belief / np.sum(belief)

    # Choose most probable state
    current_state = np.argmax(belief)

    # Select action
    if current_state == 2:
        action = "Rescue"
        reward = 20

    elif current_state == 1:
        action = random.choice(["Turn Left", "Turn Right"])
        reward = 10

    elif current_state == 0:
        action = "Move Forward"
        reward = 15

    else:
        action = random.choice(actions[:-1])
        reward = 8

    print("Sensor Observation :", states[observation])

    print("Belief State")

    for i in range(len(states)):
        print(states[i], ":", round(belief[i],3))

    print("Chosen State :", states[current_state])

    print("Selected Action :", action)

    print("Reward :", reward)

print("\n" + "=" * 75)
print("Final Belief State")
print("=" * 75)

for i in range(len(states)):
    print(states[i], ":", round(belief[i],3))

print("\nMost Probable State :", states[np.argmax(belief)])

print("\nMission Completed Successfully")