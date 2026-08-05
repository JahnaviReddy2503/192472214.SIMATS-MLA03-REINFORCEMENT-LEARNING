# Smart Traffic Signal Control Simulation

methods = ["DQN", "DDQN", "Dueling DQN", "PER"]

waiting_time = [45, 38, 32, 28]

print("Traffic Signal Performance\n")

for i in range(len(methods)):
    print(methods[i], "Waiting Time =", waiting_time[i], "seconds")

best = waiting_time.index(min(waiting_time))

print("\nBest Algorithm :", methods[best])
print("Minimum Waiting Time :", waiting_time[best], "seconds")
