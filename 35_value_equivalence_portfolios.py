import numpy as np

# "Value equivalence" here is simplified as comparing portfolios
# by predicted long-term return using historical-like synthetic data.
np.random.seed(6)

returns = np.random.normal(0.01, 0.05, (100,3))
portfolios = {
    "Conservative": np.array([0.7,0.2,0.1]),
    "Balanced": np.array([0.4,0.4,0.2]),
    "Growth": np.array([0.2,0.3,0.5])
}

for name, weights in portfolios.items():
    daily = returns @ weights
    predicted_annual = np.mean(daily) * 252
    risk = np.std(daily) * np.sqrt(252)
    score = predicted_annual - 0.5*risk
    print(f"{name:12s} Return={predicted_annual:.3f}  Risk={risk:.3f}  Score={score:.3f}")

print("\nHighest predicted score:", max(
    portfolios,
    key=lambda n: np.mean(returns @ portfolios[n])*252 - 0.5*np.std(returns @ portfolios[n])*np.sqrt(252)
))
