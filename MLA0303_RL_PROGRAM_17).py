import gymnasium as gym

# Create MountainCar environment
env = gym.make("MountainCar-v0")

# Reset environment
state, info = env.reset()

print("Starting MountainCar Simulation")

# Run for 20 steps
for i in range(20):

    # Choose a random action
    action = env.action_space.sample()

    # Perform the action
    state, reward, terminated, truncated, info = env.step(action)

    print("Step:", i + 1)
    print("Action:", action)
    print("Reward:", reward)

    if terminated or truncated:
        print("Episode Finished!")
        break

# Close environment
env.close()