import numpy as np
from cores.MC import ArrayBT
import matplotlib.pyplot as plt

# I should have coded in C++, since python is too slow. Anyway, local vol tree is just a toy.
# Mckean SDE and particle estimation are the most powerful tools in the LV model and SLV model.

class LVBT(object):
    '''
    Reference:
    http://emanuelderman.com/the-volatility-smile-and-its-implied-tree/
    '''
    def __init__(self, s0, r, y, T, Nt, sigmafunc):
        self.a = ArrayBT(s0, r, y, T, Nt)
        self.sfunc = sigmafunc
        self.sigmaT = np.zeros_like(self.a.sArray)
        self.valid = False


    def Simulation(self):
        A = self.a
        self.sigmaT[0][0] = self.sfunc(A.sArray[0][0], 0)
        A.sArray[0][1] = A.s0 * np.exp(self.sigmaT[0][0] * np.sqrt(A.dt))
        A.sArray[1][1] = A.s0 * np.exp(-self.sigmaT[0][0] * np.sqrt(A.dt))
        self.sigmaT[0][1] = self.sfunc(A.sArray[0][1], A.dt)
        self.sigmaT[1][1] = self.sfunc(A.sArray[1][1], A.dt)

        if A.Nt <= 2:
            return A.sArray

        for i in range(2, A.Nt + 1):

            for j in range(1, i+1):
                A.sArray[j][i] = A.sArray[j - 1][i - 2]
                self.sigmaT[j][i] = self.sfunc(A.sArray[j][i], A.dt * i)

            s = A.sArray[0][i - 1]
            f = s * np.exp((A.r - A.y) * A.dt)
            sigma = self.sigmaT[0][i - 1]
            A.sArray[0][i] = f + s ** 2 * sigma ** 2 * A.dt / (f - A.sArray[1][i])
            self.sigmaT[0][i] = self.sfunc(A.sArray[0][i], A.dt * i)

            s = A.sArray[i-1][i - 1]
            f = s * np.exp((A.r - A.y) * A.dt)
            sigma = self.sigmaT[i-1][i - 1]
            A.sArray[i][i] = f - s ** 2 * sigma ** 2 * A.dt / (A.sArray[i - 1][i] - f)
            self.sigmaT[i][i] = self.sfunc(A.sArray[i][i], A.dt * i)

            if self.sigmaT[0][i] <= 0 or self.sigmaT[i][i] <= 0:
                raise ValueError("Sigma < 0")

        self.valid = True
        return A.sArray


    def EuropeanPricing(self, K, type = "call"):

        if not self.valid:
            raise ValueError("No available simulation, do 'self.simulation' first")
        A = self.a
        func = np.vectorize(lambda x: max(x - K, 0) if type.lower() == "call" else max(K - x, 0))
        ep = np.zeros_like(A.sArray)
        ep[:,-1] = func(A.sArray[:,-1])

        if not hasattr(self, "RiskNeutralProb"):
            nr = self.NrProb()
        else:
            nr = self.RiskNeutralProb

        for i in range(A.Nt-1,-1,-1):
            for j in range(i+1):
                p = nr[j][i]
                ep[j][i] = (p[0] * ep[j][i+1] + p[1] * ep[j+1][i+1]) * np.exp(-A.r * A.dt)
        self.Eptree = ep

        return ep[0][0]


    def NrProb(self):

        if not self.valid:
            raise ValueError("No available path, do 'self.simulation' first")
        A = self.a
        nr = np.zeros((A.Nt + 1, A.Nt, 2))
        for i in range(A.Nt-1, -1, -1):
            for j in range(i + 1):
                s = A.sArray[j][i]
                f = s * np.exp((A.r - A.y) * A.dt)
                nr[j][i][0] = (f - A.sArray[j+1][i+1])/(A.sArray[j][i+1] - A.sArray[j+1][i+1])
                nr[j][i][1] = 1 - nr[j][i][0]
        self.RiskNeutralProb = nr

        return nr


    def show(self, figsize = (8,8), xscale = 0.5, yscale = 0.05, grid = False, add_prob = False,
             offset = 1.0, nround = 4, lw = 0.5, title = ""):

        if not self.valid:
            raise ValueError("No available path, do 'self.simulation' first")

        fig, ax = plt.subplots(1, figsize = figsize)
        A = self.a
        for i in range(A.Nt + 1):
            x = [A.dt * i]*(A.Nt + 1)
            ax.scatter(x, A.sArray[:,i])

        if grid:
            for i in range(A.Nt):
                for j in range(i+1):
                    op = (A.dt * i, A.sArray[j][i])
                    up = (A.dt * (i + 1), A.sArray[j][i + 1])
                    dp = (A.dt * (i + 1), A.sArray[j + 1][i + 1])
                    ax.plot((op[0], up[0]), (op[1], up[1]), 'k--')
                    ax.plot((op[0], dp[0]), (op[1], dp[1]), 'k--')

        if add_prob:

            if not hasattr(self, "RiskNeutralProb"):
                nr = self.NrProb()
            else:
                nr = self.RiskNeutralProb

            for i in range(A.Nt-1, -1, -1):
                for j in range(i + 1):
                    p = nr[j][i]
                    op = (A.dt * i, A.sArray[j][i])
                    up = (A.dt * (i + 1), A.sArray[j][i+1] + offset)
                    dp = (A.dt * (i + 1), A.sArray[j+1][i+1] - offset)

                    ax.annotate(round(p[0],nround), up, ((lw*op[0] + (1-lw)*up[0]),
                                                          (lw*op[1] + (1-lw)*up[1])))
                    ax.annotate(round(p[1],nround), dp, ((lw*op[0] + (1-lw)*dp[0]),
                                                          (lw*op[1] + (1-lw)*dp[1])))

        ax.set_xlim((-A.dt * xscale, A.T + A.dt * xscale))
        ax.set_ylim((A.sArray[-1][-1] * (1 - yscale), A.sArray[0][-1] * (1 + yscale)))

        if len(title) != 0:
            ax.set_title(title)

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
    lvbt.show(grid = True, add_prob=True, lw =0.7, offset = 2.5, title = "aaa")

