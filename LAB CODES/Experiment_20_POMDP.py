# Experiment 20
# Search and Rescue Robot using POMDP

states = ["Unknown", "Victim Detected", "Rescue Completed"]

belief = [0.30, 0.60, 0.90]

print("Search and Rescue Robot\n")

for i in range(len(states)):
    print("State :", states[i])
    print("Belief Probability :", belief[i])
    print()

print("Mission Completed Successfully")
