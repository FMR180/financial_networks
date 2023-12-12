import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from src.financial_network import FinancialNetwork
from src.merton_from_network import MertonFromNetwork
from src.capm_from_network import CapmFromNetwork

# network
# import liabilities matrix
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

# network
eba_network = FinancialNetwork(1, 1, external_assets, total_liabilities, pi)
eba_merton = MertonFromNetwork(eba_network, r, sigma, T)
eba_capm = CapmFromNetwork(eba_merton, r, sigma, T, beta)

# timer, since code on large network takes some time to run
starttime = datetime.now()
starttimes = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print("Process started at: ", starttimes)

# vectors to fill with data
eir_risky = np.empty(n)
eir_rf = np.empty(n)
eir_nw = np.empty(n)
mcap_risky = np.empty(n)
mcap_rf = np.empty(n)
mcap_nw = np.empty(n)
# vectors containing effective interest rate
eir_risky = eba_merton.effective_interestrate_risky()
eir_rf = eba_merton.effective_interestrate_riskfree()
eir_nw = eba_capm.effective_interest_rate()
# vectors containing market cap
mcap_risky = eba_merton.get_marketcap_risky()
mcap_rf = eba_merton.get_marketcap_riskfree()
mcap_nw = eba_capm.market_capitalization()

endtime = datetime.now()
endtimes = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print("Process finished at: ", endtimes)
print("Took: ", endtime - starttime)

# plot of the effective interest rate
plt.boxplot(eir_risky, positions=[1], labels=['Merton Risky'])
plt.boxplot(eir_rf, positions=[2], labels=['Merton Risk-Free'])
plt.boxplot(eir_nw, positions=[3], labels=['Network Effects'])
plt.ylabel('Effective Interest Rate')
plt.title('Comparison of the Effective Interest Rate')

plt.show()

# plot of the market cap
plt.boxplot(mcap_risky, positions=[1], labels=['Merton Risky'])
plt.boxplot(mcap_rf, positions=[2], labels=['Merton Risk-Free'])
plt.boxplot(mcap_nw, positions=[3], labels=['Network Effects'])
plt.ylabel('Market Capitalization (in million Euro)')
plt.title('Comparison of the Market Capitalization')

plt.show()
