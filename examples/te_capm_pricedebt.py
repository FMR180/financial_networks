import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# network
s = np.array([2, 4, 6])
pbar = np.array([10, 9, 6])
pi = np.array([[0, 1/5, 1/2], [4/9, 0, 1/3], [1/3, 1/2, 0]])
pit = pi.T
r = 0.0
T = 1.0
sigma = 1.0
qfinal = np.array([1.999, 1.09, 0.28])
lam = np.diag(np.array([1, 1, 1]))
id = np.identity(3)


def qbeta(q, beta):
    return np.exp((1 - sigma/beta)*(r + beta*sigma/2)*T)*q**(sigma/beta)


def price_debt(beta):
    z = sigma * beta
    qhat = qbeta(qfinal[2], beta)
    ds = np.diag(s)
    Delta = np.linalg.inv(np.subtract(id, np.dot(pi.T, lam)))
    delta = np.dot(np.dot(Delta, np.subtract(id, pi.T)), pbar)
    d1_hat = (np.log(1 / qhat) + (r - 1 / 2 * (sigma - 2 * z) * sigma) * T) / (sigma * np.sqrt(T))
    d2_hat = (np.log(1 / qhat) + (r - 1 / 2 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d1_nplus = (np.log(1 / 0.0000001) + (r - 1 / 2 * (sigma - 2 * z) * sigma) * T) / (sigma * np.sqrt(T))
    d2_nplus = (np.log(1 / 0.0000001) + (r - 1 / 2 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d1v = d1_hat * np.ones(3)
    d1nv = d1_nplus * np.ones(3)

    cdf1 = norm.cdf(-d1v) - norm.cdf(-d1nv)
    cdf2 = norm.cdf(-d2_hat) - norm.cdf(-d2_nplus)

    oo = np.exp(-r*T) + 1/pbar[2] * (np.dot(np.array([0, 0, 1]), np.dot(Delta, np.dot(ds, cdf1))) - np.exp(-r*T) * delta[2] * cdf2)
    return oo





x = np.linspace(0.01, 1)
b1 = np.empty(x.size)
for i in range(x.size):
    b1[i] = price_debt(x[i])





fig, ax = plt.subplots()
ax.plot(x, b1, linewidth=2.0, label="Conditional Expectation upper bound", color="tab:blue")
ax.plot(x, price_debt(1)*np.ones(x.size), linewidth=2.0, label="Comonotonic lower bound", color="tab:orange", linestyle="dashed")
ax.plot(x, price_debt(0.01)*np.ones(x.size), linewidth=2.0, label="Jensen's upper bound", color="tab:green",  linestyle="dotted")
ax.legend()
ax.set_ylabel('Price of Debt')
ax.set_xlabel("Beta \u03B2")
ax.set_title("Impact of Market Beta on Price of Debt for Bank 3")
ax.set(xlim=(0, 1.0), xticks=np.arange(0, 1, 0.2),
       ylim=(0.85, 1.005), yticks=np.arange(0.85, 1.0, .05))
plt.show()