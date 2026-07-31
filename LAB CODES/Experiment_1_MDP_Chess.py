states = ["Start", "Middle", "Winning Position"]

state = "Start"

while state != "Winning Position":

    print("Current State:", state)

    if state == "Start":
        action = "Move Pawn"
        reward = 5
        state = "Middle"

    elif state == "Middle":
        action = "Move Queen"
        reward = 20
        state = "Winning Position"

    print("Action:", action)
    print("Reward:", reward)
    print()

print("Game Won")
