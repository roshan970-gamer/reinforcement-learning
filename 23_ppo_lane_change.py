import numpy as np

# Simplified PPO-style lane-changing agent.
# State = [current_lane, traffic_slow, destination_lane]
# Actions = 0 stay, 1 left, 2 right
np.random.seed(1)
policy = np.zeros((3, 2, 3, 3))  # simple preference table
lr = 0.05
clip = 0.2

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def run_episode():
    lane = 1
    total = 0
    for _ in range(20):
        slow = np.random.randint(2)
        target = np.random.randint(3)
        probs = softmax(policy[lane, slow, target])
        action = np.random.choice(3, p=probs)

        old_lane = lane
        if action == 1 and lane > 0:
            lane -= 1
        elif action == 2 and lane < 2:
            lane += 1

        reward = 2 if lane == target else 0
        reward += 2 if slow and lane != old_lane else 0
        reward -= 5 if slow and lane == old_lane else 0
        total += reward

        advantage = reward - 0.1
        ratio = 1 + lr * advantage
        ratio = np.clip(ratio, 1-clip, 1+clip)
        policy[old_lane, slow, target, action] += lr * ratio * advantage
    return total

scores = [run_episode() for _ in range(1000)]
print("Average score:", round(np.mean(scores[-100:]), 2))
print("Policy training finished using clipped PPO-style updates.")
