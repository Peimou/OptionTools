import numpy as np
from cores import ArrayTree
import matplotlib.pyplot as plt

# I should have coded in C++, since python is too slow. Anyway, local vol tree is just a toy.
# Mckean SDE and particle estimation are the most powerful tools in the LV model and SLV model.

class LVBT(object):
    '''
    Reference:
    http://emanuelderman.com/the-volatility-smile-and-its-implied-tree/
    '''
    def __init__(self, s0, r, y, T, Nt, sigmafunc):
        self.a = ArrayTree.ArrayBT(s0, r, y, T, Nt)
        self.sfunc = sigmafunc
        self.sigmaT = np.zeros_like(self.a.sArray)
        self.valid = False


    def Simulation(self):
        self.sigmaT[0][0] = self.sfunc(self.a.sArray[0][0], 0)
        self.a.sArray[0][1] = self.a.s0 * np.exp(self.sigmaT[0][0] * np.sqrt(self.a.dt))
        self.a.sArray[1][1] = self.a.s0 * np.exp(-self.sigmaT[0][0] * np.sqrt(self.a.dt))
        self.sigmaT[0][1] = self.sfunc(self.a.sArray[0][1], self.a.dt)
        self.sigmaT[1][1] = self.sfunc(self.a.sArray[1][1], self.a.dt)

        if self.a.Nt <= 2:
            return self.a.sArray

        for i in range(2, self.a.Nt + 1):

            for j in range(1, i+1):
                self.a.sArray[j][i] = self.a.sArray[j - 1][i - 2]
                self.sigmaT[j][i] = self.sfunc(self.a.sArray[j][i], self.a.dt * i)

            s = self.a.sArray[0][i - 1]
            f = s * np.exp((self.a.r - self.a.y) * self.a.dt)
            sigma = self.sigmaT[0][i - 1]
            self.a.sArray[0][i] = f + s ** 2 * sigma ** 2 * self.a.dt / (f - self.a.sArray[1][i])
            self.sigmaT[0][i] = self.sfunc(self.a.sArray[0][i], self.a.dt * i)

            s = self.a.sArray[i-1][i - 1]
            f = s * np.exp((self.a.r - self.a.y) * self.a.dt)
            sigma = self.sigmaT[i-1][i - 1]
            self.a.sArray[i][i] = f - s ** 2 * sigma ** 2 * self.a.dt / (self.a.sArray[i - 1][i] - f)
            self.sigmaT[i][i] = self.sfunc(self.a.sArray[i][i], self.a.dt * i)

        self.valid = True
        return self.a.sArray


    def EuropeanPricing(self, K, type = "call"):
        if not self.valid:
            raise ValueError("No available simulation, do 'self.simulation' first")

        func = np.vectorize(lambda x: max(x - K, 0) if type.lower() == "call" else max(K - x, 0))
        ep = np.zeros_like(self.a.sArray)
        ep[:,-1] = func(self.a.sArray[:,-1])

        if not hasattr(self, "RiskNeutralProb"):
            nr = self.NrProb()
        else:
            nr = self.RiskNeutralProb

        for i in range(self.a.Nt-1,-1,-1):
            for j in range(i+1):
                p = nr[j][i]
                ep[j][i] = (p[0] * ep[j][i+1] + p[1] * ep[j+1][i+1]) * np.exp(-self.a.r * self.a.dt)
        self.Eptree = ep

        return ep[0][0]


    def NrProb(self):

        if not self.valid:
            raise ValueError("No available path, do 'self.simulation' first")

        nr = np.zeros((self.a.Nt + 1, self.a.Nt, 2))
        for i in range(self.a.Nt-1, -1, -1):
            for j in range(i + 1):
                s = self.a.sArray[j][i]
                f = s * np.exp((self.a.r - self.a.y) * self.a.dt)
                nr[j][i][0] = (f - self.a.sArray[j+1][i+1])/(self.a.sArray[j][i+1] - self.a.sArray[j+1][i+1])
                nr[j][i][1] = 1 - nr[j][i][0]
        self.RiskNeutralProb = nr

        return nr


    def show(self, xscale = 0.5, yscale = 0.05, grid = False, add_prob = False,
             offset = 1.0, nround = 4, lw = 0.5, title = ""):

        if not self.valid:
            raise ValueError("No available path, do 'self.simulation' first")

        for i in range(self.a.Nt + 1):
            x = [self.a.dt * i]*(self.a.Nt + 1)
            plt.scatter(x, self.a.sArray[:,i])


        if grid:
            for i in range(self.a.Nt):
                for j in range(i+1):
                    op = (self.a.dt * i, self.a.sArray[j][i])
                    up = (self.a.dt * (i + 1), self.a.sArray[j][i + 1])
                    dp = (self.a.dt * (i + 1), self.a.sArray[j + 1][i + 1])
                    plt.plot((op[0], up[0]), (op[1], up[1]), 'k--')
                    plt.plot((op[0], dp[0]), (op[1], dp[1]), 'k--')


        if add_prob:
            if hasattr(self, "RiskNeutralProb"):
                nr = self.NrProb()
            else:
                nr = self.RiskNeutralProb

            for i in range(self.a.Nt-1, -1, -1):
                for j in range(i + 1):
                    p = nr[j][i]
                    op = (self.a.dt * i, self.a.sArray[j][i])
                    up = (self.a.dt * (i + 1), self.a.sArray[j][i+1] + offset)
                    dp = (self.a.dt * (i + 1), self.a.sArray[j+1][i+1] - offset)

                    plt.annotate(round(p[0],nround), up, ((lw*op[0] + (1-lw)*up[0]),
                                                          (lw*op[1] + (1-lw)*up[1])))
                    plt.annotate(round(p[1],nround), dp, ((lw*op[0] + (1-lw)*dp[0]),
                                                          (lw*op[1] + (1-lw)*dp[1])))

        plt.xlim((-self.a.dt * xscale, self.a.T + self.a.dt * xscale))
        plt.ylim((self.a.sArray[-1][-1] * (1 - yscale), self.a.sArray[0][-1] * (1 + yscale)))

        if len(title) != 0:
            plt.title(title)

        plt.show()


#test
if __name__ == "__main__":
    s0 = 100
    r = 0.5
    y = 0
    T = 0.04
    Nt = 4

    sigmafunc = lambda x, y: max(0.2 - (x/100 - 1),0.001)
    lvbt = LVBT(s0, r, y, T, Nt, sigmafunc)
    a = lvbt.Simulation()
    b = lvbt.sigmaT
    c = lvbt.NrProb()
    print(lvbt.EuropeanPricing(98, type = "call"))
    d = lvbt.Eptree
    lvbt.show(grid = True, add_prob=True, lw =0.7, offset = 2.5)

