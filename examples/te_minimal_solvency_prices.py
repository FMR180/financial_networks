import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# network
s = np.array([2,4,6])
pbar = np.array([10,9,6])
Pi = np.array([[0, 1/5, 1/2], [4/9, 0, 1/3], [1/3, 1/2, 0]])
pit = Pi.T
r = 0.0
T = 1.0
sigma = 1.0
beta = 1.0
z = beta*sigma
q1 = 0.28
q2 = 0.28
q3 = 0.28
qfinal = np.array([1.999, 1.09, 0.28])
id = np.identity(3)
eta_hat1 = np.exp((1 - z/sigma)*(r + z*sigma/2)*T)*q1**(z/sigma)
eta_hat2 = np.exp((1 - z/sigma)*(r + z*sigma/2)*T)*q2**(z/sigma)
eta_hat3 = np.exp((1 - z/sigma)*(r + z*sigma/2)*T)*q3**(z/sigma)
eta_hat = np.array([eta_hat1, eta_hat2, eta_hat3])

lam = np.diag(np.array([1, 1, 0]))   # default
# deltas
Delta = np.linalg.inv(np.subtract(id, np.dot(pit, lam)))
delta = np.dot(np.dot(Delta, np.subtract(id, pit)), pbar)

condition1 = np.dot(Delta, np.dot(np.diag(s), eta_hat))
condition2 = delta
# condition1 < condition2
#print(condition1)
#print(condition2)

def qbeta(q, beta):
    return np.exp((1 - sigma/beta)*(r + beta*sigma/2)*T)*q**(sigma/beta)


x = np.linspace(0.01, 1)
# vectors to fill wit qbeta 1,2,3
a1 = np.empty(x.size)
a2 = np.empty(x.size)
a3 = np.empty(x.size)

for i in range(x.size):
    a1[i] = qbeta(qfinal[0], x[i])
    a2[i] = qbeta(qfinal[1], x[i])
    a3[i] = qbeta(qfinal[2], x[i])

# Plot Bank 1
fig, ax = plt.subplots()
ax.plot(x, a1, linewidth=2.0, label="Conditional Expectation Setting $q^{*}(\u03B2 \u03C3_{M})$", color="tab:blue")
ax.plot(x, qfinal[0]*np.ones(x.size), linewidth=2.0, label="Comonotonic Setting $q^{*}(\u03C3)$", color="tab:orange", linestyle="dashed")
ax.legend()
ax.set_ylabel('Minimal sSolvency Price $q^{*}$')
ax.set_xlabel("Beta \u03B2")
ax.set_title("Impact of Market Beta on Minimal Solvency Price of Bank 1")
ax.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.2),
       ylim=(0, 10), yticks=np.arange(0, 10, 2))
plt.show()

# Plot Bank 2
fig, ax = plt.subplots()
ax.plot(x, a2, linewidth=2.0, label="Conditional Expectation Setting $q^{*}(\u03B2 \u03C3_{M})$", color="tab:blue")
ax.plot(x, qfinal[1]*np.ones(x.size), linewidth=2.0, label="Comonotonic Setting $q^{*}(\u03C3)$", color="tab:orange", linestyle="dashed")
ax.legend()
ax.set_ylabel('Minimal sSolvency Price $q^{*}$')
ax.set_xlabel("Beta \u03B2")
ax.set_title("Impact of Market Beta on Minimal Solvency Price of Bank 2")
ax.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.2),
       ylim=(0, 10), yticks=np.arange(0, 10, 2))
plt.show()

# Plot Bank 3
fig, ax = plt.subplots()
ax.plot(x, a3, linewidth=2.0, label="Conditional Expectation Setting $q^{*}(\u03B2 \u03C3_{M})$", color="tab:blue")
ax.plot(x, qfinal[2]*np.ones(x.size), linewidth=2.0, label="Comonotonic Setting $q^{*}(\u03C3)$", color="tab:orange", linestyle="dashed")
ax.legend()
ax.set_ylabel('Minimal sSolvency Price $q^{*}$')
ax.set_xlabel("Beta \u03B2")
ax.set_title("Impact of Market Beta on Minimal Solvency Price of Bank 3")
ax.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.2),
       ylim=(0, 0.5), yticks=np.arange(0, 0.5, 0.1))
plt.show()