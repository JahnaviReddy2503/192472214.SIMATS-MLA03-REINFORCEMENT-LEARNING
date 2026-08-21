import numpy as np

np.random.seed(7)

STATES = 3
ACTIONS = 2
ALPHA = 0.10
GAMMA = 0.90
EPSILON = 0.20
EPISODES = 1000

Q = np.zeros((STATES,ACTIONS))

def step(state,action):
    if state == 2 and action == 1:
        return 1,10.0
    if state == 2 and action == 0:
        return 2,-10.0
    if state == 1 and action == 1:
        return 0,7.0
    if state == 1 and action == 0:
        return np.random.choice([1,2],p=[0.7,0.3]),2.0
    if state == 0 and action == 1:
        return 0,3.0
    return np.random.choice([0,1],p=[0.8,0.2]),5.0

for _ in range(EPISODES):
    state = np.random.randint(STATES)

    for _ in range(20):
        if np.random.rand() < EPSILON:
            action = np.random.randint(ACTIONS)
        else:
            action = int(np.argmax(Q[state]))

        next_state,reward = step(state,action)
        Q[state,action] += ALPHA*(
            reward + GAMMA*np.max(Q[next_state]) - Q[state,action]
        )
        state = next_state

state_names = ["Low Queue","Medium Queue","High Queue"]
action_names = ["Normal Resources","Extra Resources"]

print("="*65)
print("EXPERIMENT 39 - RL HEALTHCARE MANAGEMENT")
print("="*65)
print("\nLearned Q-table:")
print(np.round(Q,2))
print("\nLearned policy:")

for state in range(STATES):
    action = int(np.argmax(Q[state]))
    print(state_names[state],"->",action_names[action])

print("\nResult: RL healthcare resource-management policy learned successfully.")
