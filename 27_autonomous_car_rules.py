# Rule-based autonomous car on a small road network.
roads = {
    "A": ["B", "C"],
    "B": ["D"],
    "C": ["D"],
    "D": ["E"],
    "E": []
}
speed_limit = 50
safe_speed = 40

def choose_next(current, destination):
    options = roads[current]
    if destination in options:
        return destination
    return options[0] if options else current

current, destination = "A", "E"
path = [current]
while current != destination:
    nxt = choose_next(current, destination)
    print(f"At {current}: stop at intersection, check traffic, move to {nxt}")
    current = nxt
    path.append(current)

print("Safe path:", " -> ".join(path))
print("Policy: obey stop signs, stay below", speed_limit, "km/h, keep safe speed", safe_speed, "km/h")
