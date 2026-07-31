# Dynamic Programming for Taxi Routing (Value Iteration)

states = ['A', 'B', 'C', 'Destination']

rewards = {
    'A': 0,
    'B': 5,
    'C': 10,
    'Destination': 50
}

gamma = 0.9

value = {s: 0 for s in states}

for i in range(5):
    value['Destination'] = rewards['Destination']
    value['C'] = rewards['C'] + gamma * value['Destination']
    value['B'] = rewards['B'] + gamma * value['C']
    value['A'] = rewards['A'] + gamma * value['B']

print("Optimal State Values")

for s in states:
    print(s, ":", round(value[s],2))

print("\nOptimal Policy")
print("A --> B --> C --> Destination")
