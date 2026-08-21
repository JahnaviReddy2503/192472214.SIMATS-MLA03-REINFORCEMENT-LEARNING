import numpy as np

returns = np.array([
    [0.020,0.010,0.030,0.015],
    [0.010,0.020,0.010,0.020],
    [-0.010,0.030,0.020,0.010],
    [0.030,0.010,0.040,0.020],
    [0.020,0.020,0.010,0.030],
    [0.010,0.030,0.020,0.010],
    [0.020,0.010,0.030,0.020],
    [0.010,0.020,0.020,0.030]
])

portfolios = {
    "Conservative":np.array([0.40,0.30,0.20,0.10]),
    "Balanced":np.array([0.25,0.25,0.25,0.25]),
    "Growth":np.array([0.10,0.20,0.30,0.40])
}

INITIAL_VALUE = 100000.0
scores = {}

print("="*70)
print("EXPERIMENT 35 - PORTFOLIO VALUE-EQUIVALENCE ANALYSIS")
print("="*70)

for name,weights in portfolios.items():
    p_returns = returns @ weights
    mean_return = np.mean(p_returns)
    risk = np.std(p_returns)
    final_value = INITIAL_VALUE*np.prod(1+p_returns)
    score = mean_return/(risk+1e-8)
    scores[name] = score

    print("\nPortfolio:",name)
    print("Weights:",weights)
    print("Average return:",round(mean_return*100,3),"%")
    print("Risk:",round(risk*100,3),"%")
    print("Predicted final value:",round(final_value,2))
    print("Risk-adjusted score:",round(score,4))

best = max(scores,key=scores.get)
print("\nBest risk-adjusted strategy:",best)
print("Result: Portfolio strategies compared successfully.")
