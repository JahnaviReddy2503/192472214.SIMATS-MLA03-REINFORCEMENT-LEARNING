import random

# Road network
network = {
    "A": ["B", "C"],
    "B": ["D"],
    "C": ["D", "E"],
    "D": ["F"],
    "E": ["F"],
    "F": []
}

traffic_rules = {
    ("A", "B"): True,
    ("A", "C"): True,
    ("B", "D"): True,
    ("C", "D"): True,
    ("C", "E"): True,
    ("D", "F"): True,
    ("E", "F"): True
}

start = "A"
destination = "F"


def shortest_safe_path():

    queue = [(start, [start])]

    visited = set()

    while queue:

        node, path = queue.pop(0)

        if node == destination:
            return path

        if node in visited:
            continue

        visited.add(node)

        for next_node in network[node]:

            if traffic_rules.get(
                (node, next_node), False
            ):
                queue.append(
                    (next_node, path + [next_node])
                )

    return None


path = shortest_safe_path()

print("=" * 60)
print("AUTONOMOUS CAR SAFE NAVIGATION")
print("=" * 60)

print("\nStarting Point:", start)
print("Destination:", destination)
print("Traffic rules enforced: YES")

print("\nOptimal Safe Route:")
print(" -> ".join(path))

print("\nNumber of Road Segments:", len(path) - 1)

print("\nResult: Vehicle reached the destination while following safe routes.")