# Policy-Based RL Simulation

position = 0
goal = 5

print("Industrial Robotic Arm\n")

while position < goal:

    print("Current Position :", position)

    action = "Move Right"

    position += 1

    reward = 10

    print("Action :", action)
    print("Reward :", reward)
    print()

print("Pick and Place Completed Successfully")
