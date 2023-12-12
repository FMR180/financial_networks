import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# network
s = np.array([2, 4, 6])
b = np.zeros(3)
pbar = np.array([10, 9, 6])
pi = np.array([[0, 1/5, 1/2], [4/9, 0, 1/3], [1/3, 1/2, 0]])
pi14 = 1 - 1/5 - 1/2
pi24 = 1 - 4/9 - 1/3
pi34 = 1 - 1/3 - 1/2
pit = pi.T
r = 0.0
T = 1.0
sigma = 1
alpha = 0.9
qfinal = np.array([1.999, 1.09, 0.30]) #0.28
q0 = 100000000
q4 = 0.00001
qalpha = np.exp((r - sigma**2 / 2)*T + sigma * np.sqrt(T) * norm.ppf(1-alpha))
lamb0 = np.diag(np.array([0, 0, 0]))
lamb1 = np.diag(np.array([1, 0, 0]))
lamb2 = np.diag(np.array([1, 1, 0]))
lamb3 = np.diag(np.array([1, 1, 1]))
id = np.identity(3)


def qbeta(q, beta):
    return np.exp((1 - sigma/beta)*(r + beta*sigma/2)*T)*q**(sigma/beta)


def Deltas(lamb):
    Delta = np.dot(np.linalg.inv(np.subtract(id, np.dot(lamb, pi.T))), lamb)
    delta = np.dot(np.dot(-np.linalg.inv(np.subtract(id, np.dot(lamb, pi.T))), np.subtract(id, lamb)), pbar)
    return Delta, delta


def dk(q, qalpha, beta):
    d1 = (-np.log(np.minimum(qbeta(q, beta), qalpha)) + (r - 1/2*(sigma - 2*beta)*sigma)*T) / (sigma*np.sqrt(T))
    d2 = (-np.log(np.minimum(qbeta(q, beta), qalpha)) + (r - 1/2*sigma**2)*T) / (sigma*np.sqrt(T))
    return d1, d2


def ES(beta):
    # Delta_k, delta_k
    # k = 0
    Del0, del0 = Deltas(lamb0)
    Del1, del1 = Deltas(lamb1)
    Del2, del2 = Deltas(lamb2)
    Del3, del3 = Deltas(lamb3)
    d1_0, d2_0 = dk(q0, qalpha, beta)
    d1_1, d2_1 = dk(qfinal[0], qalpha, beta)
    d1_2, d2_2 = dk(qfinal[1], qalpha, beta)
    d1_3, d2_3 = dk(qfinal[2], qalpha, beta)
    d1_4, d2_4 = dk(q4, qalpha, beta)

    # A, B for each k
    A0 = np.subtract(np.dot(Del0, b*np.exp(r*T)), del0) * (norm.cdf(-d2_0) - norm.cdf(-d2_1))
    B0 = np.dot(Del0, s) * (norm.cdf(-d1_0) - norm.cdf(-d1_1))

    A1 = np.subtract(np.dot(Del1, b*np.exp(r*T)), del1) * (norm.cdf(-d2_1) - norm.cdf(-d2_2))
    B1 = np.dot(Del1, s) * (norm.cdf(-d1_1) - norm.cdf(-d1_2))

    A2 = np.subtract(np.dot(Del2, b*np.exp(r*T)), del2) * (norm.cdf(-d2_2) - norm.cdf(-d2_3))
    B2 = np.dot(Del2, s) * (norm.cdf(-d1_2) - norm.cdf(-d1_3))

    A3 = np.subtract(np.dot(Del3, b * np.exp(r * T)), del3) * (norm.cdf(-d2_3) - norm.cdf(-d2_4))
    B3 = np.dot(Del3, s) * (norm.cdf(-d1_3) - norm.cdf(-d1_4))
    #print(A0, A1, A2, A3)
    #print(B0, B1, B2, B3)
    # running sum
    # k = 0
    tr0 = pi14 * np.add(A0, B0)[0] + pi24 * np.add(A0, B0)[1] + pi34 * np.add(A0, B0)[2]
    # k = 1
    tr1 = pi14 * np.add(A1, B1)[0] + pi24 * np.add(A1, B1)[1] + pi34 * np.add(A1, B1)[2]
    # k = 2
    tr2 = pi14 * np.add(A2, B2)[0] + pi24 * np.add(A2, B2)[1] + pi34 * np.add(A2, B2)[2]
    # k = 3
    tr3 = pi14 * np.add(A3, B3)[0] + pi24 * np.add(A3, B3)[1] + pi34 * np.add(A3, B3)[2]
    ES = -1/(1-alpha) * (tr0 + tr1 + tr2 + tr3)

    return ES



kkk = ES(0.1)
#print(kkk)

# print(qbeta(qfinal[0], 0.1))
# print(qbeta(qfinal[1], 0.1))
# print(qbeta(qfinal[2], 0.1))
x = np.linspace(0.03, 1)
y = np.empty(x.size)
for i in range(x.size):
    y[i] = ES(x[i])

fig, ax = plt.subplots()
ax.plot(x, y, linewidth=2.0, label="Conditional Expectation Bound", color="tab:blue")
ax.plot(x, ES(1)*np.ones(x.size), linewidth=2.0, label="Comonotonic Bound", color="tab:orange", linestyle="dashed")
ax.plot(x, ES(0.03)*np.ones(x.size), linewidth=2.0, label="Jensen Bound", color="tab:green", linestyle="dotted")
ax.set_ylabel("Expected Shortfall")
ax.set_xlabel("Beta \u03B2")
ax.set_title("Impact of Market Beta on Scalar Systemic Risk Measure")
ax.set(xlim=(0.0, 1), xticks=np.arange(0.0, 1, 0.2),
       ylim=(-5.5, -1), yticks=np.arange(-5.5, -1, 0.5))
ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.85))
plt.show()