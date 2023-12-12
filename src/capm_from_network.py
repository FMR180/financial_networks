import math

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from src.financial_network import FinancialNetwork


class CapmFromNetwork(FinancialNetwork):
    def __init__(self, financial_network, riskfreerate, volatility, maturity, beta):
        self.network = financial_network
        super().__init__(financial_network.ax, financial_network.al, financial_network.x, financial_network.pbar, financial_network.pi)
        self.r = riskfreerate
        self.sigma = volatility
        self.T = maturity
        self.beta = beta
        self.z = self.sigma * self.beta
        self.d1list = None
        self.d2list = None
        self.min_solvency_prices = None
        self.default_order = None
        self.Deltas = None
        self.deltas = None
        self.q_vector = None

    def _default(self):
        self.beta = 1
        self.z = self.sigma
        id = np.identity(self.network.n)
        lamvec = np.zeros(self.network.n)
        lam = np.diag(lamvec)
        qvec = np.empty(self.network.n)
        qhatvec = np.empty(self.network.n)
        exclude = np.ones(self.network.n)
        vec = np.zeros(self.network.n)
        deforder = np.empty(self.network.n)
        Delta_Matrices = []
        delta_vectors = []
        d1_list = []
        d2_list = []
        d10 = (np.log(1 / 1000000) + (self.r - 1 / 2 * (self.sigma - 2 * self.z) * self.sigma) * self.T) / (self.sigma * np.sqrt(self.T)) * np.ones(self.network.n)
        d1nplus = (np.log(1 / 0.0000001) + (self.r - 1 / 2 * (self.sigma - 2 * self.z) * self.sigma) * self.T) / (self.sigma * np.sqrt(self.T)) * np.ones(
            self.network.n)
        d20 = (np.log(1 / 100000) + (self.r - 1 / 2 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2nplus = (np.log(1 / 0.0000001) + (self.r - 1 / 2 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d1_list.insert(0, d10)
        d2_list.insert(0, d20)
        Deltan = np.dot(np.linalg.inv(np.subtract(id, np.dot(np.subtract(id, (1 - self.network.al) * np.diag(np.ones(self.network.n))),
                                                             np.dot(self.network.pi.T, np.diag(np.ones(self.network.n)))))),
                        np.subtract(id, (1 - self.network.ax) * np.diag(np.ones(self.network.n))))
        deltan = np.dot(np.linalg.inv(np.subtract(id, np.dot(np.subtract(id, (1 - self.network.al) * np.diag(np.ones(self.network.n))),
                                                             np.dot(self.network.pi.T, np.diag(np.ones(self.network.n)))))),
                        np.dot(np.subtract(id, np.dot(np.subtract(id, (1 - self.network.al) * np.diag(np.ones(self.network.n))), self.network.pi.T)),
                        self.network.pbar))

        def condition(i, q):
            eta = (np.exp((1 - self.z / self.sigma) * (self.r + self.z * self.sigma / 2) * self.T) * q ** (self.z / self.sigma)) * np.ones(self.network.n)
            return np.linalg.norm(np.subtract(np.dot(Delta, np.dot(np.diag(self.network.x), eta))[i], delta[i]))

        for j in range(self.network.n):
            for i in range(self.network.n):
                Delta = np.dot(np.linalg.inv(np.subtract(id, np.dot(np.subtract(id, (1 - self.network.al) * lam),
                                                            np.dot(self.network.pi.T, lam)))),
                        np.subtract(id, (1 - self.network.ax) * lam))
                delta = np.dot(np.linalg.inv(np.subtract(id, np.dot(np.subtract(id, (1 - self.network.al) * lam),
                                                             np.dot(self.network.pi.T, lam)))),
                        np.dot(np.subtract(id, np.dot(np.subtract(id, (1 - self.network.al) * lam), self.network.pi.T)),
                        self.network.pbar))
                res1 = minimize_scalar(lambda q: condition(i, q))
                optimal_q = res1.x
                if math.isnan(optimal_q):
                    vec[i] = 0.00001
                else:
                    vec[i] = optimal_q
            vexclude = vec[exclude == 1]
            indx = np.argmax(vexclude)
            qvalue = vexclude[indx]
            indxoriginal = np.where(vec == qvalue)[0][0]
            qvec[indxoriginal] = qvalue
            qhatvec[indxoriginal] = np.exp((1 - self.sigma / self.z) * (self.r + self.z * self.sigma / 2) * self.T) * qvalue ** (self.sigma / self.z)
            exclude[indxoriginal] = 0
            Delta_Matrices.insert(indxoriginal, Delta)
            delta_vectors.insert(indxoriginal, delta)
            d1k = (np.log(1 / qhatvec[indxoriginal]) + (self.r - 1 / 2 * (self.sigma - 2 * self.z) * self.sigma) * self.T) / (
                        self.sigma * np.sqrt(self.T)) * np.ones(self.network.n)
            d1_list.insert(indxoriginal + 1, d1k)
            d2k = (np.log(1 / qhatvec[indxoriginal]) + (self.r - 1 / 2 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
            d2_list.insert(indxoriginal + 1, d2k)
            lamvec[indxoriginal] = 1
            lam = np.diag(lamvec)
            deforder[indxoriginal] = j

        d1_list.append(d1nplus)
        d2_list.append(d2nplus)
        Delta_Matrices.append(Deltan)
        delta_vectors.append(deltan)
        self.q_vector = qvec
        self.d1list = d1_list
        self.d2list = d2_list
        self.Deltas = Delta_Matrices
        self.deltas = delta_vectors
        self.default_order = deforder
        self.min_solvency_prices = qhatvec

    def _get_params(self):
        if self.d1list is None:
            self._default()
        else:
            pass

    def effective_interest_rate(self):
        self._get_params()
        reffvec = np.empty(self.network.n)
        for i in range(1, self.network.n + 1):
            sum = 0
            for k in range(i, self.network.n + 1):
                term = np.dot(self.Deltas[k], np.dot(np.diag(self.network.x), norm.cdf(-self.d1list[k]) - norm.cdf(-self.d1list[k + 1])))[
                           i - 1] - np.exp(-self.r * self.T) * self.deltas[k][i - 1] * (
                                   norm.cdf(-self.d2list[k]) - norm.cdf(-self.d2list[k + 1]))
                sum = sum + term
            debt = np.exp(-self.r * self.T) + 1 / self.network.pbar[i - 1] * sum
            reffvec[i - 1] = 1 / self.T * (np.log(self.network.pbar[i - 1]) - np.log(debt * self.network.pbar[i - 1]))
        return reffvec

    def market_capitalization(self):
        self._get_params()
        marketcapvec = np.empty(self.network.n)
        for i in range(1, self.network.n + 1):
            sume = 0
            for k in range(0, i):
                terme = np.dot(self.Deltas[k], np.dot(np.diag(self.network.x), norm.cdf(-self.d1list[k]) -
                                                      norm.cdf(-self.d1list[k + 1])))[i - 1] - np.exp(-self.r * self.T) \
                        * self.deltas[k][i - 1] * (norm.cdf(-self.d2list[k]) - norm.cdf(-self.d2list[k + 1]))
                sume = sume + terme
            equity = np.exp(-self.r * self.T) * sume
            marketcapvec[i - 1] = equity
        return marketcapvec

    def minimal_solvency_prices(self):
        self._get_params()
        return self.min_solvency_prices

    def get_default_order(self):
        self._get_params()
        return self.default_order

    def get_underlying_network(self):
        return self.network

    def get_q_vec(self):
        self._get_params()
        return self.q_vector

