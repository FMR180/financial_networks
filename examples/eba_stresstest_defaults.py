import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from src.financial_network import FinancialNetwork

# network
# import data
excel_data = pd.read_excel(r'C:\Users\Flo\Documents\Uni\Master\Masterarbeit\EBAFiles\Interbank_Liabilities.xlsx', header=None)
interbank_matrix = excel_data.to_numpy()
EL_data = pd.read_excel(r'C:\Users\Flo\Documents\Uni\Master\Masterarbeit\EBAFiles\External_Liabilities.xlsx', header=None, usecols=[0])
EL_np = EL_data.to_numpy().flatten()
TL_data = pd.read_excel(r'C:\Users\Flo\Documents\Uni\Master\Masterarbeit\EBAFiles\Total_Liabilities.xlsx', header=None, usecols=[0])
TL_np = TL_data.to_numpy().flatten()
EA_data = pd.read_excel(r'C:\Users\Flo\Documents\Uni\Master\Masterarbeit\EBAFiles\External_Assets.xlsx', header=None, usecols=[0])
EA_np = EA_data.to_numpy().flatten()
TC_data = pd.read_excel(r'C:\Users\Flo\Documents\Uni\Master\Masterarbeit\EBAFiles\Total_Capital.xlsx', header=None, usecols=[0])
TC_np = TC_data.to_numpy().flatten()
# Data of network
external_liabilities = np.array(EL_np)  # L_i,n+1
interbank_liabilities = np.sum(interbank_matrix, axis=1)    # sum L_ij
interbank_assets = interbank_liabilities    # sum L_ji = sum L_ij
total_liabilities = np.array(TL_np)     # pbar
external_assets = np.array(EA_np)      # S_i
total_capital = np.array(TC_np)     # C_i
n = external_liabilities.size
pi = interbank_matrix / total_liabilities[:, np.newaxis]    # Pi

# parameters
beta = 1
sigma = 0.27
z = beta * sigma
r = 0
T = 1

# start time
starttime = datetime.now()
starttimes = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print("Process started at: ", starttimes)

# vector of alphas to iterate
alphas = np.linspace(0, 1)

# vectors for data
vec_initial_0 = np.empty(alphas.size)
vec_initial_5 = np.empty(alphas.size)
vec_initial_75 = np.empty(alphas.size)
vec_initial_10 = np.empty(alphas.size)

vec_final_0 = np.empty(alphas.size)
vec_final_5 = np.empty(alphas.size)
vec_final_75 = np.empty(alphas.size)
vec_final_10 = np.empty(alphas.size)

for i in range(alphas.size):
    a = alphas[i]
    # create financial networks with a = ax = al and shocked initial endowments X
    ntwk0 = FinancialNetwork(a, a, external_assets, total_liabilities, pi)
    ntwk5 = FinancialNetwork(a, a, 0.95*external_assets, total_liabilities, pi)
    ntwk75 = FinancialNetwork(a, a, 0.925*external_assets, total_liabilities, pi)
    ntwk10 = FinancialNetwork(a, a, 0.9*external_assets, total_liabilities, pi)
    # fill vector of initial defaults
    vec_initial_0[i] = np.count_nonzero(ntwk0.initial_wealth() < 0)
    vec_initial_5[i] = np.count_nonzero(ntwk5.initial_wealth() < 0)
    vec_initial_75[i] = np.count_nonzero(ntwk75.initial_wealth() < 0)
    vec_initial_10[i] = np.count_nonzero(ntwk10.initial_wealth() < 0)
    # fill vector of defaults after network clearing
    vec_final_0[i] = np.count_nonzero(ntwk0.get_final_clearing_wealth() < 0)
    vec_final_5[i] = np.count_nonzero(ntwk5.get_final_clearing_wealth() < 0)
    vec_final_75[i] = np.count_nonzero(ntwk75.get_final_clearing_wealth() < 0)
    vec_final_10[i] = np.count_nonzero(ntwk10.get_final_clearing_wealth() < 0)

# end time
endtime = datetime.now()
endtimes = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print("Process finished at: ", endtimes)
print("Took: ", endtime - starttime)

# plot
fig, ax = plt.subplots()
# initial defaults
ax.plot(alphas, vec_initial_0, linewidth=2.0, label="Shock = 0.0%", color="tab:blue", linestyle="dashed")
ax.plot(alphas, vec_initial_5, linewidth=2.0, label="Shock = 5.0%", color="tab:orange", linestyle="dashed")
ax.plot(alphas, vec_initial_75, linewidth=2.0, label="Shock = 7.5%", color="tab:green", linestyle="dashed")
ax.plot(alphas, vec_initial_10, linewidth=2.0, label="Shock = 10.0%", color="tab:cyan", linestyle="dashed")
# defaults after network clearing
ax.plot(alphas, vec_final_0, linewidth=2.0, label="Shock = 0.0%", color="tab:blue", linestyle="solid")
ax.plot(alphas, vec_final_5, linewidth=2.0, label="Shock = 5.0%", color="tab:orange", linestyle="solid")
ax.plot(alphas, vec_final_75, linewidth=2.0, label="Shock = 7.5%", color="tab:green", linestyle="solid")
ax.plot(alphas, vec_final_10, linewidth=2.0, label="Shock = 10.0%", color="tab:cyan", linestyle="solid")
# legend
line_blue = plt.Line2D([0], [0], color='tab:blue', lw=2, label='Shock = 0.0%')
line_orange = plt.Line2D([0], [0], color='tab:orange', lw=2, label='Shock = 5.0%')
line_green = plt.Line2D([0], [0], color='tab:green', lw=2, label='Shock = 7.5%')
line_cyan = plt.Line2D([0], [0], color='tab:cyan', lw=2, label='Shock = 10.0%')
legend1 = ax.legend(handles=[line_blue, line_orange, line_green, line_cyan], loc="upper left", fancybox=True, framealpha=1, bbox_to_anchor=(0.02, 0.65), facecolor="white")
ax.set_xlabel("Recovery Rate \u03B1")
ax.set_ylabel("Number of defaulting Banks")
ax.set_title("Impact of Recovery Rate \u03B1 on the Number of Defaults")
plt.show()


