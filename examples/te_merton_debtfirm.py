import numpy as np
from src.financial_network import FinancialNetwork
from src.merton_from_network import MertonFromNetwork
from scipy.stats import norm
import matplotlib.pyplot as plt


class DfvrMerton(MertonFromNetwork):
    def __init__(self, merton_model):
        self.merton = merton_model
        self.network = merton_model.network
        super().__init__(merton_model.network, merton_model.r, merton_model.sigma, merton_model.T)
        self.s = self.network.x
        self.r = self.merton.r
        self.T = self.merton.T
        self.sigma = self.merton.sigma

    def r_eff_liabilities(self, d):
        pbar = self.merton.alter_liabilities(d)
        d0 = d[0]
        if d0 == 0:
            s2 = self.s[2]
        else:
            s2 = self.s[2] + 6.87
        pbar = pbar[2]
        b = 0
        c = b * np.exp(self.r * self.T) - pbar
        if c >= 0:
            EV = np.exp(-self.r * self.T)
        else:
            d_1 = (np.log(s2 / (pbar - b * np.exp(-self.r * self.T))) + (self.r + self.sigma ** 2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))
            d_2 = d_1 - self.sigma * np.sqrt(self.T)
            EV = np.exp(-self.r * self.T) + s2 / pbar * norm.cdf(-d_1) - (np.exp(-self.r * self.T) - b / pbar) * norm.cdf(-d_2)
        return -1 / self.T * np.log(EV)

    def r_eff_assets(self, d):
        d0 = d[0]
        try:
            s = self.merton.alter_assets(d)
        except:
            s = np.zeros(3)
        if d0 == 0.0:
            s2 = s[2]
            b = 8
        elif d0 == 1.25:
            s2 = s[2] + 6.87
            b = 0
        else:
            s2 = s[2] + 3.033
            b = 0
        pbar = self.pbar[2]
        c = b * np.exp(self.r * self.T) - pbar
        if c >= 0:
            EV = np.exp(-self.r * self.T)
        else:
            d_1 = (np.log(s2 / (pbar - b * np.exp(-self.r * self.T))) + (self.r + self.sigma ** 2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))
            d_2 = d_1 - self.sigma * np.sqrt(self.T)
            EV = np.exp(-self.r * self.T) + s2 / pbar * norm.cdf(-d_1) - (np.exp(-self.r * self.T) - b / pbar) * norm.cdf(-d_2)
        return -1 / self.T * np.log(EV)


# specify network
s = np.array([2, 4, 6])
pbar = np.array([10, 9, 6])
pi = np.array([[0, 1/5, 1/2], [4/9, 0, 1/3], [1/3, 1/2, 0]])
r = 0.0
sigma = 1.0
T = 1.0
b = np.array([0, 0, 0])
f_network = FinancialNetwork(1, 1, s, pbar, pi)
merton_network = MertonFromNetwork(f_network, r, sigma, T)
network = DfvrMerton(merton_network)

# create empty vectors to iterate
x = np.linspace(0, 1.5)
x2 = np.linspace(0.01, 1.2)
a1, a2, a3 = np.empty(x.size), np.empty(x.size), np.empty(x.size)
b1, b2, b3 = np.empty(x2.size), np.empty(x2.size), np.empty(x2.size)

# fill vectors with effective interest rates
for i in range(x.size):
    a1[i] = network.r_eff_liabilities(np.array([1.25, 1, x[i]]))
    a2[i] = network.r_eff_liabilities(np.array([0.8, 0.5, x[i]]))
    a3[i] = network.r_eff_liabilities(np.array([0, 0, x[i]]))

for i in range(x2.size):
    b1[i] = network.r_eff_assets(np.array([2.5, 2, x2[i]]))
    b2[i] = network.r_eff_assets(np.array([1.25, 1, x2[i]]))
    b3[i] = network.r_eff_assets(np.array([0, 0, x2[i]]))

# plot results
fig, ax = plt.subplots()
ax.plot(x, a1, linewidth=2.0, label="d1 = 1.25, d2 = 1.0", color="tab:blue", linestyle="solid")
ax.plot(x, a2, linewidth=2.0, label="d1 = 0.80, d2 = 0.5", color="tab:orange", linestyle="dashed")
ax.plot(x, a3, linewidth=2.0, label="d1 = 0.00, d2 = 0.0", color="tab:green", linestyle="dotted")
ax.set(xlim=(0, 1.5), xticks=np.arange(0, 1.5, 0.2),
       ylim=(0, 1.5), yticks=np.arange(0, 1.5, 0.5))
ax.set_xlabel("debt-firm 3 ratio")
ax.set_ylabel("effective interest rate")
ax.set_title("Effective Interest Rate for Bank 3 when changing liabilities")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(x2, b1, linewidth=2.0, label="d1 = 2.50, d2 = 2.0", color="tab:blue", linestyle="solid")
ax.plot(x2, b2, linewidth=2.0, label="d1 = 1.25, d2 = 1.0", color="tab:orange", linestyle="dashed")
ax.plot(x2, b3, linewidth=2.0, label="d1 = 0.00, d2 = 0.0", color="tab:green", linestyle="dotted")
ax.set(xlim=(0, 1.2), xticks=np.arange(0, 1.2, 0.2),
       ylim=(0, 1.0), yticks=np.arange(0, 1.0, 0.5))
ax.set_xlabel("debt-firm 3 ratio")
ax.set_ylabel("effective interest rate")
ax.set_title("Effective Interest Rate for Bank 3 when changing Assets")
ax.legend()
plt.show()