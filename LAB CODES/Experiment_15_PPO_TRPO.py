import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import random

# Parameters
state_size = 6
action_size = 4
episodes = 10
learning_rate = 0.001
clip_ratio = 0.2

# PPO Policy Network
ppo_model = Sequential([
    Dense(32, activation='relu', input_shape=(state_size,)),
    Dense(32, activation='relu'),
    Dense(action_size, activation='softmax')
])

ppo_model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy'
)

# TRPO Policy Network
trpo_model = Sequential([
    Dense(32, activation='relu', input_shape=(state_size,)),
    Dense(32, activation='relu'),
    Dense(action_size, activation='softmax')
])

trpo_model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy'
)

print("="*65)
print("PPO and TRPO for Humanoid Robot Walking")
print("="*65)

ppo_rewards = []
trpo_rewards = []

for episode in range(episodes):

    state = np.random.rand(1, state_size)

    # PPO Prediction
    ppo_probs = ppo_model.predict(state, verbose=0)[0]
    ppo_action = np.random.choice(action_size, p=ppo_probs)
    ppo_reward = random.randint(10,25)

    target = np.zeros((1, action_size))
    target[0][ppo_action] = 1

    ppo_model.fit(state, target, epochs=1, verbose=0)

    # TRPO Prediction
    trpo_probs = trpo_model.predict(state, verbose=0)[0]
    trpo_action = np.random.choice(action_size, p=trpo_probs)
    trpo_reward = random.randint(10,25)

    trpo_model.fit(state, target, epochs=1, verbose=0)

    ppo_rewards.append(ppo_reward)
    trpo_rewards.append(trpo_reward)

    print("\nEpisode :", episode+1)
    print("State :", state.round(2))
    print("PPO Action :", ppo_action, "Reward :", ppo_reward)
    print("TRPO Action:", trpo_action, "Reward :", trpo_reward)

print("\n" + "="*65)
print("Training Completed")
print("="*65)

print("\nAverage PPO Reward :", round(np.mean(ppo_rewards),2))
print("Average TRPO Reward:", round(np.mean(trpo_rewards),2))

test_state = np.random.rand(1, state_size)

ppo_policy = ppo_model.predict(test_state, verbose=0)
trpo_policy = trpo_model.predict(test_state, verbose=0)

print("\nPPO Policy")
print(ppo_policy)

print("\nTRPO Policy")
print(trpo_policy)

print("\nBest PPO Action :", np.argmax(ppo_policy))
print("Best TRPO Action:", np.argmax(trpo_policy))