import numpy as np

np.random.seed(2)
# Actions: 0=Sell, 1=Hold, 2=Buy
policy = np.zeros((3, 3))
alpha, gamma = 0.02, 0.95

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

returns = []
for episode in range(500):
    state = np.random.randint(3)
    states, actions, rewards = [], [], []

    for _ in range(20):
        p = softmax(policy[state])
        action = np.random.choice(3, p=p)
        price_change = np.random.normal(0.2, 1.0)
        reward = {-1: 0, 0: 0}.get(0, 0)

        if action == 2:
            reward = price_change
        elif action == 0:
            reward = -price_change
        else:
            reward = -0.05

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = np.random.randint(3)

    G = 0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        p = softmax(policy[states[t]])
        grad = -p
        grad[actions[t]] += 1
        policy[states[t]] += alpha * G * grad

    returns.append(sum(rewards))

print("Average final profit:", round(np.mean(returns[-100:]), 2))
print("Best action per market state:", np.argmax(policy, axis=1))
