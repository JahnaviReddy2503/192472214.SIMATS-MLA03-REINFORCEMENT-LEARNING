# Experiment 4 - Bellman Equation

gamma = 0.9

reward = [5, 10]

future = [20, 40]

probability = [0.5, 0.5]

value = 0

for p, r, f in zip(probability, reward, future):
    value += p * (r + gamma * f)

print("Optimal State Value =", value)
