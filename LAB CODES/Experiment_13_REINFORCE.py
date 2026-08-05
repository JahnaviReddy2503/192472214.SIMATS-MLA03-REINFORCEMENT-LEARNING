import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import random

# Parameters
state_size = 4
action_size = 3
episodes = 10
gamma = 0.99

# Build Policy Network
model = Sequential([
    Dense(32, activation="relu", input_shape=(state_size,)),
    Dense(32, activation="relu"),
    Dense(action_size, activation="softmax")
])

optimizer = Adam(learning_rate=0.001)

print("="*60)
print("REINFORCE Algorithm for Autonomous Parking")
print("="*60)

for episode in range(episodes):

    state = np.random.rand(1, state_size)

    with tf.GradientTape() as tape:

        # Predict action probabilities
        probs = model(state, training=True)

        probs = tf.squeeze(probs)

        # Select action according to probability distribution
        action = np.random.choice(action_size, p=probs.numpy())

        # Simulated reward
        reward = random.randint(5,20)

        # Policy Gradient Loss
        loss = -tf.math.log(probs[action]) * reward

    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print("\nEpisode :", episode+1)
    print("State :", state.round(2))
    print("Action :", action)
    print("Reward :", reward)
    print("Loss :", round(float(loss),4))

print("\nTraining Completed Successfully")

print("\nTesting Learned Policy")

test_state = np.random.rand(1,state_size)

prediction = model.predict(test_state,verbose=0)

print("\nPolicy Probabilities")

print(prediction)

print("\nBest Parking Action =",np.argmax(prediction))