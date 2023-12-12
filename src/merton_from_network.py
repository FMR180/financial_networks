import numpy as np
from scipy.stats import norm
from src.financial_network import FinancialNetwork


class MertonFromNetwork(FinancialNetwork):
    def __init__(self, financial_network, riskfreerate, volatility, maturity):
        self.network = financial_network
        super().__init__(financial_network.ax, financial_network.al, financial_network.x, financial_network.pbar, financial_network.pi)
        self.r = riskfreerate
        self.sigma = volatility
        self.T = maturity
        self.pit = self.network.pi.T

    def effective_interestrate_risky(self, T=None):
        srisky = np.add(self.network.x, np.dot(self.network.get_pitransposed(), self.network.pbar))
        brisky = np.zeros(srisky.size)
        if T is None:
            d_1 = (np.log(srisky / (self.network.pbar - brisky * np.exp(-self.r * self.T))) + (self.r + self.sigma ** 2 / 2) * self.T) / (
                        self.sigma * np.sqrt(self.T))
            d_2 = d_1 - self.sigma * np.sqrt(self.T)
            EV = np.exp(-self.r * self.T) + srisky / self.network.pbar * norm.cdf(-d_1) - (
                    np.exp(-self.r * self.T) - brisky / self.network.pbar) * norm.cdf(-d_2)
            r_eff = -1 / self.T * np.log(EV)
        else:
            d_1 = (np.log(srisky / (self.network.pbar - brisky * np.exp(-self.r * T))) + (
                        self.r + self.sigma ** 2 / 2) * T) / (
                          self.sigma * np.sqrt(T))
            d_2 = d_1 - self.sigma * np.sqrt(T)
            EV = np.exp(-self.r * T) + srisky / self.network.pbar * norm.cdf(-d_1) - (
                    np.exp(-self.r * T) - brisky / self.network.pbar) * norm.cdf(-d_2)
            r_eff = -1 / T * np.log(EV)
        return r_eff

    def effective_interestrate_riskfree(self, T=None):
        srf = self.network.x
        brf = np.dot(self.network.get_pitransposed(), self.network.pbar)
        r_eff = np.empty(srf.size)
        if T is None:
            c = brf * np.exp(self.r * self.T) - self.network.pbar
            for i in range(c.size):
                if c[i] >= 0:
                    EV = np.exp(self.r * self.T)
                else:
                    d_1 = (np.log(srf[i] / (self.network.pbar[i] - brf[i] * np.exp(-self.r * self.T))) + (
                            self.r + self.sigma ** 2 / 2) * self.T) / (
                                  self.sigma * np.sqrt(self.T))
                    d_2 = d_1 - self.sigma * np.sqrt(self.T)
                    EV = np.exp(-self.r * self.T) + srf[i] / self.network.pbar[i] * norm.cdf(-d_1) - (
                           np.exp(-self.r * self.T) - brf[i] / self.network.pbar[i]) * norm.cdf(-d_2)
                r_eff[i] = -1 / self.T * np.log(EV)
        else:
            c = brf * np.exp(self.r * T) - self.network.pbar
            for i in range(c.size):
                if c[i] >= 0:
                    EV = np.exp(self.r * T)
                else:
                    d_1 = (np.log(srf[i] / (self.network.pbar[i] - brf[i] * np.exp(-self.r * T))) + (
                            self.r + self.sigma ** 2 / 2) * T) / (
                                  self.sigma * np.sqrt(T))
                    d_2 = d_1 - self.sigma * np.sqrt(T)
                    EV = np.exp(-self.r * T) + srf[i] / self.network.pbar[i] * norm.cdf(-d_1) - (
                            np.exp(-self.r * T) - brf[i] / self.network.pbar[i]) * norm.cdf(-d_2)
                r_eff[i] = -1 / T * np.log(EV)
        return r_eff

    def get_marketcap_risky(self, T=None):
        if T is None:
            T = self.T
        else:
            T = T
        srisky = np.add(self.network.x, np.dot(self.network.get_pitransposed(), self.network.pbar))
        brisky = np.zeros(srisky.size)
        EV = np.empty(srisky.size)
        c = brisky * np.exp(self.r * T) - self.network.pbar
        for i in range(c.size):
            if c[i] >= 0:
                EV[i] = brisky[i] + srisky[i] - np.exp(-self.r*T)*self.network.pbar[i]
            else:
                d_1 = (np.log(srisky[i] / (self.network.pbar[i] - brisky[i] * np.exp(-self.r * T))) + (self.r + self.sigma**2 / 2) * T) / (self.sigma * np.sqrt(T))
                d_2 = d_1 - self.sigma * np.sqrt(T)
                EV[i] = srisky[i] * norm.cdf(d_1) - (self.network.pbar[i]*np.exp(-self.r*T) - brisky[i])*norm.cdf(d_2)
        return EV

    def get_marketcap_riskfree(self, T=None):
        if T is None:
            T = self.T
        else:
            T = T
        srf = self.network.x
        brf = np.dot(self.network.get_pitransposed(), self.network.pbar)
        EV = np.empty(srf.size)
        c = brf * np.exp(self.r * T) - self.network.pbar
        for i in range(c.size):
            if c[i] >= 0:
                EV[i] = brf[i] + srf[i] - np.exp(-self.r*T)*self.network.pbar[i]
            else:
                d_1 = (np.log(srf[i] / (self.network.pbar[i] - brf[i] * np.exp(-self.r * T))) + (self.r + self.sigma**2 / 2) * T) / (self.sigma * np.sqrt(T))
                d_2 = d_1 - self.sigma * np.sqrt(T)
                EV[i] = srf[i] * norm.cdf(d_1) - (self.network.pbar[i]*np.exp(-self.r*T) - brf[i])*norm.cdf(d_2)
        return EV

    def get_debt_firm_ratio(self):
        dfr = np.dot(np.linalg.inv(np.diag(np.add(self.network.x, np.dot(self.pit, self.network.pbar)))), self.network.pbar)
        return dfr

    def alter_assets(self, d):
        diagd = np.diag(d)
        mat = np.linalg.inv(diagd)
        vec = np.subtract(self.network.pbar, np.dot(np.dot(diagd, self.pit), self.network.pbar))
        s = np.dot(mat, vec)
        return s

    def alter_liabilities(self, d):
        diagd = np.diag(d)
        idm = np.identity(d.size)
        mat = np.linalg.inv(np.subtract(idm, np.dot(diagd, self.pit)))
        vec = np.dot(diagd, self.network.x)
        pbar = np.dot(mat, vec)
        return pbar

# s = np.array([3, 4])
# pbar = np.array([10, 6])
# L = np.array([[0, 7], [3, 0]])
# pi = np.array([[0, 7/10], [1/2, 0]])
# network = FinancialNetwork(1,1,s,pbar,pi)
# merton = MertonFromNetwork(network, 0, 1, 1)
#
# print(merton.effective_interestrate_riskfree())