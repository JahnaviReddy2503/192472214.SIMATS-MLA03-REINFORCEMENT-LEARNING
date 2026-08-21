import numpy as np

np.random.seed(7)

STATES = 3
ACTIONS = 3
ALPHA = 0.10
GAMMA = 0.90
EPSILON = 0.20
EPISODES = 1000

Q = np.zeros((STATES,ACTIONS))

def step(state,action):
    if state == 0:
        if action == 0:
            return 1,8.0
        if action == 1:
            return 1,4.0
        return 0,-5.0

    if state == 1:
        if action == 0:
            return 1,3.0
        if action == 1:
            return 2,8.0
        return 2,4.0

    if action == 2:
        return 2,10.0
    return 2,5.0

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

state_names = ["Beginner","Intermediate","Advanced"]
lesson_names = ["Easy Lesson","Normal Lesson","Advanced Lesson"]

print("="*65)
print("EXPERIMENT 40 - RL PERSONALIZED EDUCATION")
print("="*65)
print("\nLearned Q-table:")
print(np.round(Q,2))
print("\nPersonalized learning policy:")

for state in range(STATES):
    action = int(np.argmax(Q[state]))
    print(state_names[state],"->",lesson_names[action])

print("\nResult: Personalized educational policy learned successfully.")
