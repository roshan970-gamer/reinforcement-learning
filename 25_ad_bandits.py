import numpy as np

np.random.seed(3)
true_ctr = np.array([0.05, 0.10, 0.07, 0.15, 0.09])
N = len(true_ctr)
T = 5000

def epsilon_greedy(eps=0.1):
    counts = np.zeros(N)
    rewards = np.zeros(N)
    for _ in range(T):
        a = np.random.randint(N) if np.random.rand() < eps else np.argmax(rewards / np.maximum(counts, 1))
        r = np.random.rand() < true_ctr[a]
        counts[a] += 1
        rewards[a] += r
    return rewards.sum() / T

def ucb():
    counts = np.ones(N)
    rewards = np.zeros(N)
    for a in range(N):
        rewards[a] += np.random.rand() < true_ctr[a]
    for t in range(N, T):
        mean = rewards / counts
        score = mean + np.sqrt(2 * np.log(t + 1) / counts)
        a = np.argmax(score)
        r = np.random.rand() < true_ctr[a]
        counts[a] += 1
        rewards[a] += r
    return rewards.sum() / T

def thompson():
    success = np.ones(N)
    failure = np.ones(N)
    for _ in range(T):
        samples = np.random.beta(success, failure)
        a = np.argmax(samples)
        r = np.random.rand() < true_ctr[a]
        success[a] += r
        failure[a] += 1 - r
    return (success.sum() - N) / T

results = {"Epsilon-Greedy": epsilon_greedy(), "UCB": ucb(), "Thompson": thompson()}
for k, v in results.items():
    print(f"{k}: CTR = {v:.4f}")
print("Best:", max(results, key=results.get))
