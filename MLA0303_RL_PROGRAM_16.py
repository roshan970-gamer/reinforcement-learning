# Grid (3x3)
# S = Start, G = Goal, X = Obstacle

grid = [
    ["S", ".", "."],
    [".", "X", "."],
    [".", ".", "G"]
]

# State values
value = [
    [0, 0, 0],
    [0, -1, 0],   # -1 represents obstacle
    [0, 0, 10]    # Goal value = 10
]

# Bellman Update (Value Iteration)
for k in range(5):
    for i in range(3):
        for j in range(3):

            # Skip obstacle and goal
            if grid[i][j] == "X" or grid[i][j] == "G":
                continue

            neighbors = []

            # Up
            if i > 0 and value[i-1][j] != -1:
                neighbors.append(value[i-1][j])

            # Down
            if i < 2 and value[i+1][j] != -1:
                neighbors.append(value[i+1][j])

            # Left
            if j > 0 and value[i][j-1] != -1:
                neighbors.append(value[i][j-1])

            # Right
            if j < 2 and value[i][j+1] != -1:
                neighbors.append(value[i][j+1])

            # Update value
            if neighbors:
                value[i][j] = max(neighbors) - 1

# Print State Value Function
print("State Value Function:")
for row in value:
    print(row)

# Print Optimal Path
print("\nOptimal Path:")
print("(0,0) → (0,1) → (0,2) → (1,2) → (2,2)")