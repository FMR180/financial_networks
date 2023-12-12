import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from src.financial_network import FinancialNetwork
from src.merton_from_network import MertonFromNetwork
from src.capm_from_network import CapmFromNetwork
from src.systemic_risk_measures import SystemicRiskMeasure

# network
s = np.array([2, 4, 6])
b = np.zeros(3)
pbar = np.array([10, 9, 6])
pi = np.array([[0, 1/5, 1/2], [4/9, 0, 1/3], [1/3, 1/2, 0]])
soc = np.array([3, 2, 1])
r = 0.0
T = 1.0
sigma = 1
alpha = 0.9
beta = 0.1
x = np.linspace(0.01, 1)
y = np.empty(x.size)
z = np.empty(x.size)
zz = np.empty(x.size)
network = FinancialNetwork(1, 1, s, pbar, pi)
capm = CapmFromNetwork(network, r, sigma, T, beta)
print(network.initial_wealth())
print(network.get_final_clearing_wealth())
for i in range(x.size):
    zz[i] = SystemicRiskMeasure(capm).cvar_societal_wealth(alpha, x[i], soc)
    y[i] = SystemicRiskMeasure(capm).cvar_number_full_paying(alpha, x[i])
    z[i] = SystemicRiskMeasure(capm).cvar_system_wealth(alpha, x[i])




# Plot Bank 3
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=2.0, label="Conditional Expectation Bound", color="tab:blue")
ax.plot(x, SystemicRiskMeasure(capm).cvar_number_full_paying(alpha, 0.01)*np.ones(x.size), linewidth=2.0, linestyle='dotted', label="Jensen Bound", color="tab:green")
ax.plot(x, SystemicRiskMeasure(capm).cvar_number_full_paying(alpha, 1)*np.ones(x.size), linewidth=2.0, linestyle='dashed', label="Comonotonic Bound", color="tab:orange")
ax.legend()
ax.set_ylabel('Expected Shortfall')
ax.set_xlabel("Beta \u03B2")
ax.set_title("Impact of Market Beta on Scalar Systemic Risk Measure")
plt.show()

# Plot Bank 3
fig, ax = plt.subplots()
ax.plot(x, z, linewidth=2.0, label="Conditional Expectation Bound", color="tab:blue")
ax.plot(x, SystemicRiskMeasure(capm).cvar_system_wealth(alpha, 0.01)*np.ones(x.size), linewidth=2.0, linestyle='dotted', label="Jensen Bound", color="tab:green")
ax.plot(x, SystemicRiskMeasure(capm).cvar_system_wealth(alpha, 1)*np.ones(x.size), linewidth=2.0, linestyle='dashed', label="Comonotonic Bound", color="tab:orange")
ax.legend()
ax.set_ylabel('Expected Shortfall')
ax.set_xlabel("Beta \u03B2")
ax.set_title("Impact of Market Beta on Scalar Systemic Risk Measure")
plt.show()

