# Experiment 2 - RL Smart Home Robot

position = 0
goal = 5

while position < goal:

    print("\nCurrent Position:", position)

    action = "Move Forward"

    position += 1

    reward = 10

    print("Action:", action)
    print("Reward:", reward)
    print("Next Position:", position)

print("\nDestination Reached Successfully!")
