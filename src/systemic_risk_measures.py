from src.financial_network import FinancialNetwork
from src.capm_from_network import CapmFromNetwork
import numpy as np
from scipy.stats import norm


class SystemicRiskMeasure(CapmFromNetwork):
    def __init__(self, capm):
        self.capm = capm
        self.network = self.capm.network
        self.pbar = self.capm.pbar
        self.r = self.capm.r
        self.n = self.capm.n
        self.T = self.capm.T
        self.beta = 1
        self.sigma = self.capm.sigma
        self.x = self.capm.x
        self.pi = self.capm.pi
        self.z = self.capm.z
        self.ax = self.capm.ax
        self.al = self.capm.al
        self.d1list = None
        self.d2list = None
        self.min_solvency_prices = None
        self.default_order = None
        self.Deltas = None
        self.deltas = None
        self.q_vector = None

    def cvar_societal_wealth(self, alpha, beta, societal_liabilities_nominal):
        pi_soc = societal_liabilities_nominal / self.pbar
        b = np.zeros(self.n)
        qalpha = np.exp((self.r - self.sigma ** 2 / 2) * self.T + self.sigma * np.sqrt(self.T) * norm.ppf(1 - alpha))
        deflist = self.get_default_order()
        qvec = self.minimal_solvency_prices()
        qbeta_vec = np.exp((1 - self.sigma/beta)*(self.r + beta*self.sigma/2)*self.T)*qvec**(self.sigma/beta)
        lambvec = np.zeros(self.n)
        lamb = np.diag(lambvec)
        dgo = np.diag(np.ones(self.n))
        id = np.identity(self.n)
        Delta_list = []
        delta_list = []
        d10 = (-np.log(qalpha) + (self.r - 1/2*(self.sigma - 2*beta)*self.sigma)*self.T) / (self.sigma*np.sqrt(self.T))
        d20 = (-np.log(qalpha) + (self.r - 1 / 2 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d1nplus = (-np.log(0.00001) + (self.r - 1/2*(self.sigma - 2*beta)*self.sigma)*self.T) / (self.sigma*np.sqrt(self.T))
        d2nplus = (-np.log(0.00001) + (self.r - 1 / 2 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d1_list = []
        d2_list = []
        d1_list.insert(0, d10)
        d2_list.insert(0, d20)
        Deltan = self.ax*np.dot(np.linalg.inv(np.subtract(id, np.dot(self.al*dgo, self.pi.T))), dgo)
        deltan = np.dot(np.dot(-np.linalg.inv(np.subtract(id, np.dot(self.al*dgo, self.pi.T))), np.subtract(id, dgo)), self.pbar)
        Alist = []
        Blist = []
        # Fill List of Deltas and d's
        for i in range(self.n):
            index = np.where(deflist == i)[0][0]
            Delta = self.ax*np.dot(np.linalg.inv(np.subtract(id, np.dot(self.al*lamb, self.pi.T))), lamb)
            delta = np.dot(np.dot(-np.linalg.inv(np.subtract(id, np.dot(self.al*lamb, self.pi.T))), np.subtract(id, lamb)), self.pbar)
            Delta_list.insert(index, Delta)
            delta_list.insert(index, delta)
            q = qbeta_vec[index]
            d1 = (-np.log(np.minimum(q, qalpha)) + (self.r - 1/2*(self.sigma - 2*beta)*self.sigma)*self.T) / (self.sigma*np.sqrt(self.T))
            d2 = (-np.log(np.minimum(q, qalpha)) + (self.r - 1 / 2 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
            d1_list.insert(index+1, d1)
            d2_list.insert(index+1, d2)
            lambvec[index] = 1
            lamb = np.diag(lambvec)
        d1_list.append(d1nplus)
        d2_list.append(d2nplus)
        Delta_list.append(Deltan)
        delta_list.append(deltan)

        # shifted default list
        shifted_def_list = np.zeros(len(deflist) + 1, dtype=int)
        shifted_def_list[1:] = deflist+1

        # calculate A and B
        A0 = np.subtract(np.dot(Delta_list[0], b*np.exp(self.r*self.T)), delta_list[0]) * (norm.cdf(-d2_list[0]) - norm.cdf(-d2_list[1]))
        B0 = np.dot(Delta_list[0], self.x) * (norm.cdf(-d1_list[0]) - norm.cdf(-d1_list[1]))
        Alist.insert(0, A0)
        Blist.insert(0, B0)

        for k in range(self.n):
            index = np.where(shifted_def_list == k+1)[0][0]
            if k+1 >= self.n:
                indexplus = self.n+1
            else:
                indexplus = np.where(shifted_def_list == (k+2))[0][0]
            A = np.subtract(np.dot(Delta_list[index], b*np.exp(self.r*self.T)), delta_list[index]) * (norm.cdf(-d2_list[index]) - norm.cdf(-d2_list[indexplus]))
            B = np.dot(Delta_list[index], self.x) * (norm.cdf(-d1_list[index]) - norm.cdf(-d1_list[indexplus]))
            Alist.insert(index, A)
            Blist.insert(index, B)
        # calculate sums
        sum2 = 0
        for k in range(self.n+1):
            sum = 0
            for i in range(self.n):
                index = np.where(deflist == i)[0][0]
                terms = pi_soc[index] * np.add(Alist[k], Blist[k])[index]
                sum = sum + terms
            sum2 = sum2 + sum
        # calculate ES
        ES = -1/(1 - alpha) * sum2
        return ES

    def cvar_number_full_paying(self, alpha, beta):
        qalpha = np.exp((self.r - self.sigma ** 2 / 2) * self.T + self.sigma * np.sqrt(self.T) * norm.ppf(1 - alpha))
        deflist = self.get_default_order()
        qvec = self.minimal_solvency_prices()
        qbeta_vec = np.exp((1 - self.sigma / beta) * (self.r + beta * self.sigma / 2) * self.T) * qvec ** (
                    self.sigma / beta)
        d20 = (-np.log(qalpha) + (self.r - 1 / 2 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2nplus = (-np.log(0.00001) + (self.r - 1 / 2 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2_list = []
        d2_list.insert(0, d20)
        # Fill List of Deltas and d's
        for i in range(self.n):
            index = np.where(deflist == i)[0][0]
            q = qbeta_vec[index]
            d2 = (-np.log(np.minimum(q, qalpha)) + (self.r - 1 / 2 * self.sigma ** 2) * self.T) / (
                        self.sigma * np.sqrt(self.T))
            d2_list.insert(index + 1, d2)
        d2_list.append(d2nplus)

        # shifted default list
        shifted_def_list = np.zeros(len(deflist) + 1, dtype=int)
        shifted_def_list[1:] = deflist + 1
        bracket_terms_list = []
        for k in range(self.n):
            index = np.where(shifted_def_list == k + 1)[0][0]
            if k + 1 >= self.n:
                indexplus = self.n + 1
            else:
                indexplus = np.where(shifted_def_list == (k + 2))[0][0]
            A = (norm.cdf(-d2_list[index]) - norm.cdf(-d2_list[indexplus]))
            bracket_terms_list.insert(index, A)
        # calculate sums
        sum2 = 0
        #print(bracket_terms_list)
        for k in range(self.n):
            terms = (self.n - k) * bracket_terms_list[k]
            sum2 += terms
        # calculate ES
        ES = -1 / (1 - alpha) * sum2
        return ES

    def cvar_system_wealth(self, alpha, beta):  # LambN
        b = np.zeros(self.n)
        qalpha = np.exp((self.r - self.sigma ** 2 / 2) * self.T + self.sigma * np.sqrt(self.T) * norm.ppf(1 - alpha))
        deflist = self.get_default_order()
        qvec = self.minimal_solvency_prices()
        qbeta_vec = np.exp((1 - self.sigma / beta) * (self.r + beta * self.sigma / 2) * self.T) * qvec ** (
                    self.sigma / beta)
        lambvec = np.zeros(self.n)
        lamb = np.diag(lambvec)
        dgo = np.diag(np.ones(self.n))
        id = np.identity(self.n)
        Delta_list = []
        delta_list = []
        d10 = (-np.log(qalpha) + (self.r - 1 / 2 * (self.sigma - 2 * beta) * self.sigma) * self.T) / (
                    self.sigma * np.sqrt(self.T))
        d20 = (-np.log(qalpha) + (self.r - 1 / 2 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d1nplus = (-np.log(0.00001) + (self.r - 1 / 2 * (self.sigma - 2 * beta) * self.sigma) * self.T) / (
                    self.sigma * np.sqrt(self.T))
        d2nplus = (-np.log(0.00001) + (self.r - 1 / 2 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d1_list = []
        d2_list = []
        d1_list.insert(0, d10)
        d2_list.insert(0, d20)
        Deltan = self.ax * np.dot(np.linalg.inv(np.subtract(id, np.dot(self.al * dgo, self.pi.T))), dgo)
        deltan = np.dot(np.dot(-np.linalg.inv(np.subtract(id, np.dot(self.al * dgo, self.pi.T))), np.subtract(id, dgo)),
                        self.pbar)
        Alist = []
        Blist = []
        # Fill List of Deltas and d's
        for i in range(self.n):
            index = np.where(deflist == i)[0][0]
            Delta = self.ax * np.dot(np.linalg.inv(np.subtract(id, np.dot(self.al * lamb, self.pi.T))), lamb)
            delta = np.dot(
                np.dot(-np.linalg.inv(np.subtract(id, np.dot(self.al * lamb, self.pi.T))), np.subtract(id, lamb)),
                self.pbar)
            Delta_list.insert(index, Delta)
            delta_list.insert(index, delta)
            q = qbeta_vec[index]
            d1 = (-np.log(np.minimum(q, qalpha)) + (self.r - 1 / 2 * (self.sigma - 2 * beta) * self.sigma) * self.T) / (
                        self.sigma * np.sqrt(self.T))
            d2 = (-np.log(np.minimum(q, qalpha)) + (self.r - 1 / 2 * self.sigma ** 2) * self.T) / (
                        self.sigma * np.sqrt(self.T))
            d1_list.insert(index + 1, d1)
            d2_list.insert(index + 1, d2)
            lambvec[index] = 1
            lamb = np.diag(lambvec)
        d1_list.append(d1nplus)
        d2_list.append(d2nplus)
        Delta_list.append(Deltan)
        delta_list.append(deltan)

        # shifted default list
        shifted_def_list = np.zeros(len(deflist) + 1, dtype=int)
        shifted_def_list[1:] = deflist + 1

        # calculate A and B
        A0 = np.subtract(np.dot(Delta_list[0], b * np.exp(self.r * self.T)), delta_list[0]) * (
                    norm.cdf(-d2_list[0]) - norm.cdf(-d2_list[1]))
        B0 = np.dot(Delta_list[0], self.x) * (norm.cdf(-d1_list[0]) - norm.cdf(-d1_list[1]))
        Alist.insert(0, A0)
        Blist.insert(0, B0)

        for k in range(self.n):
            index = np.where(shifted_def_list == k + 1)[0][0]
            if k + 1 >= self.n:
                indexplus = self.n + 1
            else:
                indexplus = np.where(shifted_def_list == (k + 2))[0][0]
            A = np.subtract(np.dot(Delta_list[index], b * np.exp(self.r * self.T)), delta_list[index]) * (
                        norm.cdf(-d2_list[index]) - norm.cdf(-d2_list[indexplus]))
            B = np.dot(Delta_list[index], self.x) * (norm.cdf(-d1_list[index]) - norm.cdf(-d1_list[indexplus]))
            Alist.insert(index, A)
            Blist.insert(index, B)
        # calculate sums
        sum2 = 0
        for k in range(self.n + 1):
            terms = np.add(np.add(np.dot(np.subtract(b * np.exp(self.r * self.T), self.pbar),(norm.cdf(-d2_list[k]) - norm.cdf(-d2_list[k+1]))),
                           np.dot(self.x, (norm.cdf(-d1_list[k]) - norm.cdf(-d1_list[k+1])))), np.add(Alist[k], Blist[k]))
            add_terms = np.dot(np.ones(terms.size), terms)
            sum2 += add_terms
        # calculate ES
        ES = -1 / (1 - alpha) * sum2
        return ES




