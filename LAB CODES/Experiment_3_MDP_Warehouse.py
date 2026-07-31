# Experiment 3 - MDP Warehouse Robot

states = ["Start", "Pick Area", "Delivery Area", "Goal"]

current_state = "Start"

while current_state != "Goal":

    print("\nCurrent State:", current_state)

    if current_state == "Start":
        action = "Move to Pick Area"
        reward = 10
        current_state = "Pick Area"

    elif current_state == "Pick Area":
        action = "Pick Item"
        reward = 20
        current_state = "Delivery Area"

    elif current_state == "Delivery Area":
        action = "Deliver Item"
        reward = 30
        current_state = "Goal"

    print("Action:", action)
    print("Reward:", reward)
    print("Next State:", current_state)

print("\nDelivery Completed Successfully!")
