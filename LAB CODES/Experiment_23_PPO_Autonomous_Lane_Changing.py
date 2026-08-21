import numpy as np
import tensorflow as tf
import random

# Autonomous Highway Lane Changing using PPO

STATE_SIZE = 3
ACTION_SIZE = 3

# Actions
# 0 = Stay in Lane
# 1 = Change to Left Lane
# 2 = Change to Right Lane

# Actor Network
actor = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(STATE_SIZE,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(ACTION_SIZE, activation="softmax")
])

# Critic Network
critic = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(STATE_SIZE,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
])

actor_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)

critic_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)

clip_ratio = 0.2
episodes = 20

print("=" * 60)
print("PPO AUTONOMOUS HIGHWAY LANE CHANGING")
print("=" * 60)

for episode in range(episodes):

    # State:
    # [current lane, traffic speed, distance to destination]

    state = np.array([[
        random.uniform(0, 2),
        random.uniform(0.3, 1.0),
        random.uniform(0.1, 1.0)
    ]], dtype=np.float32)

    old_probabilities = actor(state).numpy()[0]

    action = np.random.choice(
        ACTION_SIZE,
        p=old_probabilities
    )

    # Simulated environment reward

    traffic_speed = state[0][1]

    if action == 0:
        reward = traffic_speed * 5

    elif action == 1:
        reward = traffic_speed * 7

    else:
        reward = traffic_speed * 7

    # Advantage estimate
    advantage = np.array(
        [[reward]],
        dtype=np.float32
    )

    # PPO Actor Update

    with tf.GradientTape() as tape:

        probabilities = actor(
            state,
            training=True
        )

        new_probability = probabilities[0, action]

        old_probability = tf.constant(
            old_probabilities[action],
            dtype=tf.float32
        )

        ratio = (
            new_probability /
            (old_probability + 1e-8)
        )

        clipped_ratio = tf.clip_by_value(
            ratio,
            1 - clip_ratio,
            1 + clip_ratio
        )

        objective1 = ratio * advantage[0, 0]

        objective2 = (
            clipped_ratio *
            advantage[0, 0]
        )

        actor_loss = -tf.minimum(
            objective1,
            objective2
        )

    actor_gradients = tape.gradient(
        actor_loss,
        actor.trainable_variables
    )

    actor_optimizer.apply_gradients(
        zip(
            actor_gradients,
            actor.trainable_variables
        )
    )

    # Critic Update

    with tf.GradientTape() as tape:

        value = critic(
            state,
            training=True
        )

        critic_loss = tf.reduce_mean(
            tf.square(
                advantage - value
            )
        )

    critic_gradients = tape.gradient(
        critic_loss,
        critic.trainable_variables
    )

    critic_optimizer.apply_gradients(
        zip(
            critic_gradients,
            critic.trainable_variables
        )
    )

    print(
        "Episode:",
        episode + 1,
        "Action:",
        action,
        "Reward:",
        round(float(reward), 2)
    )


print("\nTraining completed successfully.")

# Test the trained policy

test_state = np.array(
    [[1.0, 0.8, 0.5]],
    dtype=np.float32
)

policy = actor(
    test_state
).numpy()[0]

print("\nTest State:")
print(
    "[Current Lane, Traffic Speed, Distance]"
)

print(test_state[0])

print("\nPPO Action Probabilities:")
print(np.round(policy, 3))

best_action = np.argmax(policy)

action_names = [
    "Stay in Lane",
    "Change to Left Lane",
    "Change to Right Lane"
]

print(
    "\nRecommended Action:",
    action_names[best_action]
)

print("\nResult:")
print(
    "PPO-based lane-changing policy "
    "was successfully trained and evaluated."
)