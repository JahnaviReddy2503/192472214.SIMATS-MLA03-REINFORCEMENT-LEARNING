import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import random

# Parameters
state_size = 4
action_size = 3
episodes = 10
learning_rate = 0.001

# Shared Input
inputs = Input(shape=(state_size,))

# Hidden Layers
x = Dense(32, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)

# Actor Output
actor = Dense(action_size, activation='softmax', name="Actor")(x)

# Critic Output
critic = Dense(1, activation='linear', name="Critic")(x)

# Build Model
model = Model(inputs=inputs, outputs=[actor, critic])

optimizer = Adam(learning_rate=learning_rate)

print("="*60)
print("Actor-Critic (A2C / A3C) Elevator Scheduling")
print("="*60)

for episode in range(episodes):

    state = np.random.rand(1, state_size)

    with tf.GradientTape() as tape:

        action_probs, state_value = model(state, training=True)

        action_probs = tf.squeeze(action_probs)

        action = np.random.choice(action_size, p=action_probs.numpy())

        reward = random.randint(5,20)

        advantage = reward - state_value[0][0]

        actor_loss = -tf.math.log(action_probs[action]) * advantage

        critic_loss = tf.square(advantage)

        total_loss = actor_loss + critic_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print("\nEpisode :", episode+1)
    print("Current Floor State :", state.round(2))
    print("Selected Elevator Action :", action)
    print("Reward :", reward)
    print("State Value :", round(float(state_value[0][0]),2))

print("\nTraining Completed Successfully")

print("\nTesting Learned Policy")

test_state = np.random.rand(1,state_size)

policy,value = model.predict(test_state,verbose=0)

print("\nActor Probabilities")

print(policy)

print("\nCritic State Value")

print(value)

print("\nBest Elevator Action =",np.argmax(policy))