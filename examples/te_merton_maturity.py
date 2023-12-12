import numpy as np
from src.financial_network import FinancialNetwork
from src.merton_from_network import MertonFromNetwork
from src.capm_from_network import CapmFromNetwork
from scipy.stats import norm
import matplotlib.pyplot as plt

# specify network
s = np.array([2, 4, 6])
pbar = np.array([10, 9, 6])
pi = np.array([[0, 1/5, 1/2], [4/9, 0, 1/3], [1/3, 1/2, 0]])
r = 0.0
sigma = 1
T = 1.0
b = np.array([0, 0, 0])
f_network = FinancialNetwork(1, 1, s, pbar, pi)
merton_network = MertonFromNetwork(f_network, r, sigma, T)

# create vector of maturities to iterate over
x = np.linspace(0.1, 15)
# vector to fill with interest rates and market caps
# risky approach
b1_r = np.empty(x.size)
b2_r = np.empty(x.size)
b3_r = np.empty(x.size)
b1m_r = np.empty(x.size)
b2m_r = np.empty(x.size)
b3m_r = np.empty(x.size)
# risk-free approach
b1_rf = np.empty(x.size)
b2_rf = np.empty(x.size)
b3_rf = np.empty(x.size)
b1m_rf = np.empty(x.size)
b2m_rf = np.empty(x.size)
b3m_rf = np.empty(x.size)
# network effects
b1_nw = np.empty(x.size)
b2_nw = np.empty(x.size)
b3_nw = np.empty(x.size)
b1m_nw = np.empty(x.size)
b2m_nw = np.empty(x.size)
b3m_nw = np.empty(x.size)

# loop
for i in range(x.size):
    b1_r[i] = merton_network.effective_interestrate_risky(x[i])[0]
    b2_r[i] = merton_network.effective_interestrate_risky(x[i])[1]
    b3_r[i] = merton_network.effective_interestrate_risky(x[i])[2]
    b1m_r[i] = merton_network.get_marketcap_risky(x[i])[0]
    b2m_r[i] = merton_network.get_marketcap_risky(x[i])[1]
    b3m_r[i] = merton_network.get_marketcap_risky(x[i])[2]

    b1_rf[i] = merton_network.effective_interestrate_riskfree(x[i])[0]
    b2_rf[i] = merton_network.effective_interestrate_riskfree(x[i])[1]
    b3_rf[i] = merton_network.effective_interestrate_riskfree(x[i])[2]
    b1m_rf[i] = merton_network.get_marketcap_riskfree(x[i])[0]
    b2m_rf[i] = merton_network.get_marketcap_riskfree(x[i])[1]
    b3m_rf[i] = merton_network.get_marketcap_riskfree(x[i])[2]

    b1_nw[i] = CapmFromNetwork(f_network, r, sigma, x[i], 1).effective_interest_rate()[0]
    b2_nw[i] = CapmFromNetwork(f_network, r, sigma, x[i], 1).effective_interest_rate()[1]
    b3_nw[i] = CapmFromNetwork(f_network, r, sigma, x[i], 1).effective_interest_rate()[2]
    b1m_nw[i] = CapmFromNetwork(f_network, r, sigma, x[i], 1).market_capitalization()[0]
    b2m_nw[i] = CapmFromNetwork(f_network, r, sigma, x[i], 1).market_capitalization()[1]
    b3m_nw[i] = CapmFromNetwork(f_network, r, sigma, x[i], 1).market_capitalization()[2]



# Plots
# Bank 1
fig, ax = plt.subplots()
a1 = ax.plot(x, b1_nw, linewidth=2.0, label="Network Effect", color="tab:blue")
a2 = ax.plot(x, b1_r, linewidth=2.0, label="Merton: risky", linestyle="dashed", color="tab:blue")
a3 = ax.plot(x, b1_rf, linewidth=2.0, label="Merton: risk-free", linestyle="dotted", color="tab:blue")
ax2 = ax.twinx()
l2 = ax2.plot(x, b1m_nw, label="Network Effect", color="tab:orange")
l3 = ax2.plot(x, b1m_r, label="Merton: risky", linestyle="dashed", color="tab:orange")
l4 = ax2.plot(x, b1m_rf, label="Merton: risk-free", linestyle="dotted", color="tab:orange")
black_line_solid = plt.Line2D([0], [0], color='black', lw=2, label='Network Effect')
black_line_dashed = plt.Line2D([0], [0], linestyle="dashed", color='black', lw=2, label='Merton: risky')
black_line_dotted = plt.Line2D([0], [0], linestyle="dotted" ,color='black', lw=2, label='Merton: risk-free')
legend1 = ax.legend(handles=[black_line_solid, black_line_dashed, black_line_dotted], loc="upper left", fancybox=True, framealpha=1, bbox_to_anchor=(0.02, 0.98), facecolor="white")
ax.spines["left"].set_color("tab:blue")
ax.yaxis.label.set_color("tab:blue")
ax.tick_params(axis="y", colors="tab:blue")
ax.set_ylabel("Effective Interest Rate")
ax.set_xlabel("Maturity T")
ax.set_title("Impact of Maturity on Bank 1")
ax.set(xlim=(0, 16), xticks=np.arange(0, 15, 5),
       ylim=(0, 2), yticks=np.arange(0, 2, 0.25))
ax2.spines["right"].set_color("tab:orange")
ax2.yaxis.label.set_color("tab:orange")
ax2.tick_params(axis="y", colors="tab:orange")
ax2.set_ylabel("Market Capitalization")
ax2.set(ylim=(0, 9), yticks=np.arange(0, 9, 2))
plt.show()

# Bank 2
fig2, bx = plt.subplots()
bx.plot(x, b2_nw, linewidth=2.0, label="Network Effect", color="tab:blue")
bx.plot(x, b2_r, linewidth=2.0, label="Merton: risky", linestyle="dashed", color="tab:blue")
bx.plot(x, b2_rf, linewidth=2.0, label="Merton: risk-free", linestyle="dotted", color="tab:blue")
bx2 = bx.twinx()
lb2, = bx2.plot(x, b2m_nw, label="Network Effect", color="tab:orange")
lb2, = bx2.plot(x, b2m_r, label="Merton: risky", linestyle="dashed", color="tab:orange")
lb2, = bx2.plot(x, b2m_rf, label="Merton: risk-free", linestyle="dotted", color="tab:orange")
black_line_solid = plt.Line2D([0], [0], color='black', lw=2, label='Network Effect')
black_line_dashed = plt.Line2D([0], [0], linestyle="dashed", color='black', lw=2, label='Merton: risky')
black_line_dotted = plt.Line2D([0], [0], linestyle="dotted" ,color='black', lw=2, label='Merton: risk-free')
legend1 = bx.legend(handles=[black_line_solid, black_line_dashed, black_line_dotted], loc="upper left", fancybox=True, framealpha=1, bbox_to_anchor=(0.02, 0.98), facecolor="white")
bx.set_ylabel("Effective Interest Rate")
bx.set_xlabel("Maturity T")
bx.set_title("Impact of Maturity on Bank 2")
bx.spines["left"].set_color("tab:blue")
bx.yaxis.label.set_color("tab:blue")
bx.tick_params(axis="y", colors="tab:blue")
bx.set(xlim=(0, 16), xticks=np.arange(0, 15, 5),
       ylim=(0, 2), yticks=np.arange(0, 2, 0.25))
bx2.spines["right"].set_color("tab:orange")
bx2.yaxis.label.set_color("tab:orange")
bx2.tick_params(axis="y", colors="tab:orange")
bx2.set_ylabel("Market Capitalization")
bx2.set(ylim=(0, 11))
plt.show()

# Bank 3
fig3, cx = plt.subplots()
c1 = cx.plot(x, b3_nw, linewidth=2.0, label="Network Effect", color="tab:blue")
c2 = cx.plot(x, b3_r, linewidth=2.0, label="Merton: risky", linestyle="dashed", color="tab:blue")
c3 = cx.plot(x, b3_rf, linewidth=2.0, label="Merton: risk-free", linestyle="dotted", color="tab:blue")
cx2 = cx.twinx()
lc2 = cx2.plot(x, b3m_nw, label="Network Effect", color="tab:orange")
lc3 = cx2.plot(x, b3m_r, label="Merton: risky", linestyle="dashed", color="tab:orange")
lc4 = cx2.plot(x, b3m_rf, label="Merton: risk-free", linestyle="dotted", color="tab:orange")
black_line_solid = plt.Line2D([0], [0], color='black', lw=2, label='Network Effect')
black_line_dashed = plt.Line2D([0], [0], linestyle="dashed", color='black', lw=2, label='Merton: risky')
black_line_dotted = plt.Line2D([0], [0], linestyle="dotted" ,color='black', lw=2, label='Merton: risk-free')
legend2 = cx.legend(handles=[black_line_solid, black_line_dashed, black_line_dotted], loc="lower right", fancybox=True, framealpha=1, facecolor="white")
cx.spines["left"].set_color("tab:blue")
cx.yaxis.label.set_color("tab:blue")
cx.tick_params(axis="y", colors="tab:blue")
cx.set_ylabel("Effective Interest Rate")
cx.set_xlabel("Maturity T")
cx.set_title("Impact of Maturity on Bank 3")
cx.set(xlim=(0, 16), xticks=np.arange(0, 15, 5),
       ylim=(0, 0.25), yticks=np.arange(0, 0.25, 0.05))
cx2.spines["right"].set_color("tab:orange")
cx2.yaxis.label.set_color("tab:orange")
cx2.tick_params(axis="y", colors="tab:orange")
cx2.set_ylabel("Market Capitalization")
cx2.set(ylim=(0, 15))
plt.show()