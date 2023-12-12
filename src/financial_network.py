import numpy as np


class FinancialNetwork:

    def __init__(self, alphax, alphal, endowments, pbar, pi):
        self.ax = alphax
        self.al = alphal
        self.x = endowments
        self.pbar = pbar
        self.pi = pi
        self.pit = pi.T
        self.n = self.x.size

    def initial_wealth(self):
        return np.subtract(np.add(self.x, np.dot(self.pit, self.pbar)), self.pbar)

    def get_final_clearing_wealth(self):
        k, iterations, s_defaulting = 0, 0, 0
        Lambda = np.zeros((self.n, self.n))
        Id = np.identity(self.n)
        z = np.zeros(self.n)
        z2 = np.zeros(self.n)
        V = self.initial_wealth()
        initial_wealth = V

        while k < self.n:
            if np.any(V < 0):
                defaultingBanks = V < 0
                z2[defaultingBanks] = 1

            if np.any(z != z2):
                k = k + 1
                Lambda = np.diag(z2)
                # calculate new V
                V = np.dot(np.linalg.inv(Id - np.dot((Id - (1.0 - self.ax) * Lambda), np.dot(self.pi.T, Lambda))),
                           np.subtract(np.add(np.dot(Id - (1 - self.al) * Lambda, self.x), np.dot((Id - (1.0 - self.al) * Lambda),
                                                                                        np.dot(self.pi.T, self.pbar))),
                                       self.pbar))
            else:
                iterations = k
                k = self.n
            z = z2
        s_defaulting = int(np.sum(z2))
        return V

    def get_alphax(self):
        return self.ax

    def get_alphal(self):
        return self.al

    def get_x(self):
        return self.x

    def get_pbar(self):
        return self.pbar

    def get_pi(self):
        return self.pi

    def get_pitransposed(self):
        return self.pit

    def get_dimension(self):
        return self.n






