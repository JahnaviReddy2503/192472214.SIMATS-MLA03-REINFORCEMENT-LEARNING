print("Deep Q-Network (DQN) Simulation\n")

layers = [
    ("Input Layer", 4),
    ("Hidden Layer 1", 24),
    ("Hidden Layer 2", 24),
    ("Output Layer", 2)
]

for name, neurons in layers:
    print(f"{name} : {neurons} neurons")

print("\nOptimizer : Adam")
print("Loss Function : Mean Squared Error")
print("\nDQN Model Created Successfully")
