import numpy as np

np.random.seed(4)
prices = np.array([50, 60, 70, 80, 90])
probability = np.array([0.90, 0.78, 0.60, 0.45, 0.30])
T = 3000
N = len(prices)

def demand(a):
    return np.random.rand() < probability[a]

def epsilon_greedy():
    count = np.zeros(N)
    revenue = np.zeros(N)
    for _ in range(T):
        a = np.random.randint(N) if np.random.rand() < 0.1 else np.argmax(revenue / np.maximum(count, 1))
        r = prices[a] if demand(a) else 0
        count[a] += 1
        revenue[a] += r
    return revenue.sum()

def ucb():
    count = np.ones(N)
    revenue = np.array([prices[a] * demand(a) for a in range(N)], dtype=float)
    for t in range(N, T):
        mean = revenue / count
        score = mean + 50 * np.sqrt(2*np.log(t+1)/count)
        a = np.argmax(score)
        r = prices[a] if demand(a) else 0
        count[a] += 1
        revenue[a] += r
    return revenue.sum()

def thompson():
    success = np.ones(N)
    failure = np.ones(N)
    total = 0
    for _ in range(T):
        p = np.random.beta(success, failure)
        a = np.argmax(prices * p)
        sale = demand(a)
        total += prices[a] if sale else 0
        success[a] += sale
        failure[a] += 1 - sale
    return total

results = {"Epsilon-Greedy": epsilon_greedy(), "UCB": ucb(), "Thompson Sampling": thompson()}
for k, v in results.items():
    print(f"{k}: Total revenue = {v:.2f}")
print("Best:", max(results, key=results.get))
