import numpy as np
from cores import ArrayTree
import matplotlib.pyplot as plt

# I should have coded in C++, since python is too slow. Anyway, local vol tree is just a toy.
# Mckean SDE and particle estimation are the most powerful tools in the LV model and SLV model.

class LVBT(object):
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
            raise ValueError("No available simulation")

        func = np.vectorize(lambda x: max(x - K, 0) if type.lower() == "call" else max(K - x, 0))
        ep = np.zeros_like(self.a.sArray)
        ep[:,-1] = func(self.a.sArray[:,-1])

        for i in range(self.a.Nt-1,-1,-1):
            for j in range(i+1):
                s = self.a.sArray[j][i]
                f = s * np.exp((self.a.r - self.a.y) * self.a.dt)
                p = (f - self.a.sArray[j+1][i+1])/(self.a.sArray[j][i+1] - self.a.sArray[j+1][i+1])
                ep[j][i] = (p * ep[j][i+1] + (1-p) * ep[j+1][i+1]) * np.exp(-self.a.r * self.a.dt)

        self.Eptree = ep
        return ep[0][0]


    def show(self, limscale = 0.05):

        if not self.valid:
            raise ValueError("No available simulation")

        for i in range(self.a.Nt + 1):
            x = [self.a.dt * i]*(self.a.Nt + 1)
            plt.scatter(x, self.a.sArray[:,i])
        plt.ylim((self.a.sArray[-1][-1] * (1 - limscale), self.a.sArray[0][-1] * (1 + limscale)))
        plt.show()


#test
if __name__ == "__main__":
    s0 = 100
    r = 0
    y = 0
    T = 0.05
    Nt = 100

    sigmafunc = lambda x, y: max(0.2 - (x/100 - 1),0.001)
    lvbt = LVBT(s0, r, y, T, Nt, sigmafunc)
    a = lvbt.Simulation()
    b = lvbt.sigmaT

    print(lvbt.EuropeanPricing(100, type = "call"))
    d = lvbt.Eptree
    lvbt.show()

