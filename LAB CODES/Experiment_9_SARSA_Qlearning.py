alpha = 0.5
gamma = 0.9

Q = [[0,0],
     [0,0]]

reward = 10

print("Initial Q Table")

for row in Q:
    print(row)

print("\nQ-Learning Update")

Q[0][1] = Q[0][1] + alpha*(reward + gamma*max(Q[1]) - Q[0][1])

Q[1][1] = Q[1][1] + alpha*(20 + gamma*max(Q[1]) - Q[1][1])

for row in Q:
    print(row)

print("\nSARSA Update")

Q[0][0] = Q[0][0] + alpha*(5 + gamma*Q[1][1] - Q[0][0])

for row in Q:
    print(row)
