"""
Stochastic Volatility model
"""

import numpy as np
import numba as nb
from cores.MC import MC_Path

class Heston(object):
    def __init__(self, s0, r, y, vol0, volsigma, targetvol,
                alpha, rho, T, Nt, Npath, Vmodel = "CIR", neg = True, thres = 1e-5):
        """
        The basic model of Heston is:
        dS = rSdt + np.sqrt(sigma)*S*dZ
        d sigma = alpha(m - sigma)dt + xi np.sqrt(sigma) dw (CIR process)
        dzdw = rho

        Sometimes, people also use mean reverting GBM to model stochastic volatility.
        d sigma = alpha(m - sigma)dt + xi sigma dw (OU process)

        Parameters
        ----------
        s0: The initial price
        r: Constant interest rate.
        y: Constant dividend yield.
        vol0: The initial volatility.
        volsigma: The volatility of volatility
        targetvol: The mean of volatility
        alpha: The reversion coefficients
        rho: Correlation coefficients of the price process and volatility process.
        T: Time to maturity
        Nt: Number of time slots
        Npath: Number of Paths
        neg: Whether use covariate method to reduce the variance of MC
        thres: The threshold of OU process.

        Reference
        ---------
        https://en.wikipedia.org/wiki/Heston_model
        """

        self.s0 = s0
        self.r = r
        self.y = y
        self.pmu = r - y
        self.vol0 = vol0
        self.volsigma = volsigma
        self.targetvol = targetvol
        self.alpha = alpha
        self.rho = rho
        self.T = T
        self.Nt = Nt
        self.Npath = Npath
        self.neg = neg
        self.thres = thres
        self.Vmodel = Vmodel

    @staticmethod
    def genCorGuassian(cov_mat, Nt, Npath, neg = True, tol = 1e-8, rs = None):
        Nvar = len(cov_mat)
        u,s,v = np.linalg.svd(cov_mat)
        psd = u @ np.diag(s) @ u.T

        if Npath%2 != 0 and neg:
            raise ValueError("Npath must be even if covariate variance reduction  method is used")
        if not np.allclose(psd, cov_mat):
            raise ValueError("Need positive semi-definite covariance matrix")

        s[s<tol] = 0
        L = u * np.sqrt(s)
        if not rs:
            np.random.seed(rs)
        if neg:
            cg = np.tile(np.random.normal(size = (Nvar, int(0.5 * Npath) * Nt)), (1,2))
        else:
            cg = np.random.normal(size=(Nvar, Npath * Nt))
        cg = L @ cg
        return cg.reshape(Nvar, Npath, Nt)

    @staticmethod
    @nb.njit()
    def OUprocess(s0, alpha, mu, sigma, T, path, thres = 1e-5):
        Npath, Nt = path.shape
        Nt -= 1
        path[:, 0] = s0
        dt = T / Nt
        for i in range(Nt):
            path[:, i + 1] = alpha * mu * dt + (1 - alpha * dt ) * path[:, i] \
                             + sigma * np.sqrt(dt) * path[:, i + 1]
            path[:, i + 1][path[:, i + 1] < thres] = thres
        return path

    @staticmethod
    @nb.njit()
    def GBMMRprocess(s0, alpha, mu, sigma, T, path, thres=1e-5):
        Npath, Nt = path.shape
        Nt -= 1
        path[:, 0] = s0
        dt = T / Nt
        for i in range(Nt):
            path[:, i + 1] = alpha * mu * dt + (1 - alpha * dt + sigma * np.sqrt(dt)
                                                * path[:, i + 1]) * path[:, i]

            path[:, i + 1][path[:, i + 1] < thres] = thres
        return path

    @staticmethod
    @nb.njit()
    def CIRprocess(s0, alpha, mu, sigma, T, path, thres=1e-5):
        Npath, Nt = path.shape
        Nt -= 1
        path[:, 0] = s0
        dt = T / Nt
        for i in range(Nt):
            path[:, i + 1] = alpha * mu * dt + (1 - alpha * dt) * path[:, i] + \
                             sigma * np.sqrt(dt) * path[:, i + 1] * np.sqrt(path[:, i])

            path[:, i + 1][path[:, i + 1] < thres] = thres
        return path

    @classmethod
    def SV_Path(cls, s0, pmu, vol0, volsigma, targetvol,
                alpha, rho, T, Nt, Npath, Vmodel,
                neg = True, thres = 1e-5):
        dt = T/Nt
        cov = np.diag(np.ones(2)) + np.diag([rho], -1) + np.diag([rho], 1)
        srv, vrv = cls.genCorGuassian(cov, Nt, Npath, neg)
        if Vmodel.lower() == "ou":
            volpath = cls.OUprocess(vol0, alpha, targetvol, volsigma, T, vrv, thres)
        elif Vmodel.lower() == "cir":
            volpath = cls.CIRprocess(vol0, alpha, targetvol, volsigma, T, vrv, thres)
        elif Vmodel.lower() == "gbmmr":
            volpath = cls.GBMMRprocess(vol0, alpha, targetvol, volsigma, T, vrv, thres)
        else:
            raise ValueError(f"Volatility model {Vmodel} is not available. Currently, "
                             f"I only support 'GBMMR', 'OU', and 'CIR'.")
        srv = srv * volpath * np.sqrt(dt) - 0.5 * np.power(volpath, 2) * dt + pmu * dt
        spath = np.hstack([np.full((Npath, 1), s0), np.exp(srv)])
        spath = np.cumprod(spath, axis=1)
        return MC_Path(s0, pmu, volpath, T, Nt, Npath, spath)


    def Simulate(self):
        self._path = self.SV_Path(self.s0, self.pmu, self.vol0, self.volsigma,
                            self.targetvol,self.alpha, self.rho, self.T,
                            self.Nt, self.Npath, self.Vmodel, self.neg, self.thres)
        return self._path


    def MC_EuropeanPricing(self, K, otype = "Call"):
        if not hasattr(self, "_path"):
            raise ValueError("No available path, .Simulate() first.")
        s = self._path.Path[:, -1] - K
        if otype.lower() == "put":
            s *= -1
        return np.mean(s), np.var(s)/len(s)


if __name__ == "__main__":
    print("Testing:")
    # Multivariate gen test
    # he = Heston()
    # cor = np.array([[1,0.5],[0.5,1]])
    # Nt = 10000
    # Npath = 2
    # a=he.genCorGuassian(cor, Nt, Npath)
    # print(np.cov(a[0][0],a[1][0]))
    # print(np.cov(a[0][1], a[1][1])) #pass

    # Path generation
    s0, r, y = 100, 0.05, 0.01
    vol0, volsigma = 0.3, 1
    targetvol = 0.3
    alpha, rho, T = 1, 0.4, 1
    Nt = 100
    Npath = 100000
    he = Heston( s0, r, y, vol0, volsigma, targetvol,
                alpha, rho, T, Nt, Npath)
    a = he.Simulate()
    print(he.MC_EuropeanPricing(K = 100))


