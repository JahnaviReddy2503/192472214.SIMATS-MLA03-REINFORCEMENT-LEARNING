print("Actor-Critic Algorithm Comparison\n")

algorithms = ["A2C", "A3C"]

waiting_time = [20,16]

for i in range(len(algorithms)):

    print(algorithms[i], "Average Waiting Time =", waiting_time[i], "seconds")

best = waiting_time.index(min(waiting_time))

print("\nBest Algorithm :", algorithms[best])
