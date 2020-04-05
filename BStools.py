import numpy as np
from numeric import NewtonRaphson as nr
from math import erf, pi
from cores.MC import MC_Path
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('error')


class BS():
    def __init__(self,s = None, k = None, r = None, sigma = None, T = None, type = None, **config):
        self.S = s
        self.K = k
        self.r = r
        self.sigma = sigma
        self.T = T
        self.type = type
        self.y = 0
        self.price = None

        for item in config.items():
            if hasattr(self, item[0]):
                setattr(self, item[0], item[1])


    @staticmethod
    def ncdf(x):
        return (1.0 + erf(x / np.sqrt(2.0))) / 2.0


    @staticmethod
    def npdf(x):
        return 1/np.sqrt(2 * pi) * np.exp(- np.power(x, 2) / 2)

    @staticmethod
    def calc_d1(S, K, r, y, sigma, T):
        if T != 0:
            d1 = (np.log(S / K) + (r - y + np.power(sigma, 2) / 2) * T) / (sigma * np.sqrt(T))
        else:
            d1 = 1e6*np.sign(S-K)
        return d1


    @classmethod
    @np.vectorize
    def calc_bsprice(cls, S, K, r, y, sigma, T, type = "call"):
        d1 = cls.calc_d1(S, K, r, y, sigma, T)
        d2 = d1 - sigma * np.sqrt(T)
        if type.lower() == "call":
            return S * np.exp(-y * T) * cls.ncdf(d1) - K * np.exp(- r * T) * cls.ncdf(d2)
        if type.lower() == "put":
            return K * np.exp(- r * T) * cls.ncdf(- d2) - S * np.exp(-y * T) * cls.ncdf(- d1)
        else:
            raise TypeError(f"No such kind of option: {type}")


    def bsprice(self):
        return self.calc_bsprice(self.S, self.K, self.r, self.y, self.sigma, self.T, self.type)


    @classmethod
    @np.vectorize
    def calc_vega(cls, S, K, r, y, sigma, T, type = "call"):
        d1 = cls.calc_d1(S, K, r, y, sigma, T)
        return S * np.exp(-y * T) * cls.npdf(d1) * np.sqrt(T)


    def vega(self):
        return self.calc_vega(self.S, self.K, self.r, self.y, self.sigma, self.T)


    @classmethod
    @np.vectorize
    def calc_delta(cls, S, K, r, y, sigma, T, type):
        d1 = cls.calc_d1(S, K, r, y, sigma, T)
        if type.lower() == "call":
            return cls.ncdf(d1) * np.exp(-y * T)
        if type.lower() == "put":
            return (cls.ncdf(d1) - 1) * np.exp(-y * T)
        else:
            raise TypeError(f"No such kind of option: {type}")


    def delta(self):
        return self.calc_delta(self.S, self.K, self.r, self.y, self.sigma, self.T, self.type)


    @classmethod
    @np.vectorize
    def calc_gamma(cls, S, K, r, y, sigma, T, type = "call"):
        d1 = cls.calc_d1(S, K, r, y, sigma, T)
        return cls.npdf(d1) * np.exp(-y * T) / (S * sigma * np.sqrt(T))


    def gamma(self):
        return self.calc_gamma(self.S, self.K, self.r, self.y, self.sigma, self.T)


    @classmethod
    @np.vectorize
    def calc_theta(cls, S, K, r, y, sigma, T, type):
        d1 = cls.calc_d1(S, K, r, y, sigma, T)
        d2 = d1 - sigma * np.sqrt(T)
        if type.lower() == "call":
            return - S * cls.npdf(d1) * sigma * np.exp(-y * T) / (2 * np.sqrt(T)) \
                   + y * S * np.exp(-y * T) * cls.ncdf(d1) - r * K * np.exp(-r * T) * cls.ncdf(d2)
        if type.lower() == "put":
            return - S * cls.npdf(d1) * sigma * np.exp(-y * T) / (2 * np.sqrt(T)) \
                   + y * S * np.exp(-y * T) * cls.ncdf(d1) - r * K * np.exp(-r * T) * cls.ncdf(d2)
        else:
            raise TypeError(f"No such kind of option: {type}")


    def theta(self):
        return self.calc_theta(self.S, self.K, self.r, self.y, self.sigma, self.T, self.type)


    @classmethod
    @np.vectorize
    def calc_implied_vol(cls, price, S, K, r, y, T, type, nb = False):
        # Don't input sigma in this formula
        f1 = lambda x: cls.calc_bsprice(S, K, r, y, x, T, type) - price
        f2 = lambda x: cls.calc_vega(S, K, r, y, x, T)

        sigma = 0.5

        if nb:
            nt = nr.Newton(sigma, f1, f2)
            return nt.iteration()
        else:
            return nr.iteration(sigma, f1, f2, 1e-5, 10000)


    def implied_vol(self, nb = False):
        return self.calc_implied_vol(self.price, self.S, self.K, self.r, self.y, self.T, self.type, nb)


    @classmethod
    @np.vectorize
    def calc_log_moneyness(cls, S, K, r, y, T):
        return np.log(K) - np.log(S * np.exp((r - y) * T))


    def log_moneyness(self):
        return self.calc_log_moneyness(self.S, self.K, self.r, self.y, self.T)


    #implied Volatility model
    @classmethod
    @np.vectorize
    def SVI_surface(cls, S, K, r, y, T, a, b, c, rho, theta):
        m = cls.calc_log_moneyness(S, K, r, y, T)
        return a + b * (rho* (m - c) + np.sqrt(np.power(m - c, 2) + np.power(theta, 2)))


    @classmethod
    def Gaussian_Path(cls, s0, mu, sigma, T, N_path, N_t, method = "Euler", seed = None):
        if seed != None:
            np.random.seed(seed)
        dt = T/N_t
        if method.lower() not in ["euler", "c"]:
            raise TypeError(f"No method called {method}, try 'Euler'' or 'C'")
        mu = mu if method.lower() == "euler" else mu - 0.5 * sigma**2
        rand = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt), size=(N_path, N_t))
        path = np.hstack([np.full_like(rand[:,0],s0).reshape(-1,1), np.exp(rand)])
        path = np.cumprod(path, axis = 1)
        p = MC_Path(s0, mu, sigma, T, N_t, N_path, path)
        return p


    @classmethod
    def Plot_path(cls, path_cls, T, N_t, title = ""):
        axis_x = np.linspace(0,T,N_t + 1)
        path = path_cls.Path
        for con in path:
            plt.plot(axis_x, con)
        plt.xlabel("Time")
        plt.ylabel("Price")
        if len(title) == 0:
            plt.title(f" {path.shape[0]} Gaussian Paths: mu({path_cls.mu}), sigma({path_cls.sigma})")
        else:
            plt.title(title)
        return plt

    @classmethod
    def Delta_Hedge_Vol_PnL(cls, Path:MC_Path, S, K, r, y, hedge_vol):
        #hedge vol is the implied volatility
        path = Path.Path
        if S!= path[0][0]:
            Path.Path *= S/path[0][0]
        dt = Path.T/Path.N_t
        dpath = np.diff(path, axis = 1)
        taxis = np.arange(Path.N_t)+1
        disf = np.exp(- r * taxis * dt)

        def calc_ss(s):
            ss = np.power(hedge_vol * s[:-1], 2) * dt
            return ss

        def calc_gamma(s):
            return cls.calc_gamma(s[:-1], K, r, y, hedge_vol, np.linspace(Path.T, 0, Path.N_t, False))

        gamma = np.apply_along_axis(calc_gamma, arr=path, axis=1)
        ss_path = np.apply_along_axis(calc_ss, arr=path, axis=1)
        hedge_path = 0.5 * np.sum(gamma * (np.power(dpath, 2) - ss_path) * disf, axis=1)
        return hedge_path


    @classmethod
    def Replication_Pricing(cls, Path:MC_Path, S, K, r, y, hedge_vol, type, trans_cost = 0.0):
        path = Path.Path
        if S != path[0][0]:
            Path.Path *= S / path[0][0]
        dt = Path.T / Path.N_t
        taxis = np.arange(Path.N_t) +1
        disf = np.exp(- r * taxis * dt)
        T = Path.T

        def calc_delta(s):
            return cls.calc_delta(s[:-1], K, r, y, hedge_vol, np.linspace(Path.T, 0, Path.N_t, False), type)

        delta_path = np.apply_along_axis(calc_delta, arr = path, axis = 1)
        d_s = np.diff(delta_path, axis = 1) * path[:,1:path.shape[1]-1] * disf[:-1]
        trc = np.abs(d_s) * (1 + trans_cost) * disf[:-1]
        div = (np.exp(y * dt) - 1)*delta_path*path[:,:-1]*disf
        payoff_func = np.vectorize(lambda x: max(x-K, 0) if type.lower() == "call" else max(K-x, 0))
        payoff = payoff_func(path[:,-1])
        price = delta_path[:,0]*S + np.sum(d_s, axis = 1) - np.sum(div, axis = 1) + np.exp(-r*T)*payoff
        price -= delta_path[:, -1]*path[:,-1]*np.exp(-r*T) + trc
        return price



if __name__ == "__main__":
    bs = BS()
    mu = 0
    sigma = 0.3
    T = 1
    r = 0.01
    y = 0.02
    S = 100
    K = 100
    path = bs.Gaussian_Path(S, mu, sigma, T, 100, 20, "C", 50)
    c = bs.Replication_Pricing(path, S, K, r, y, 0.4, "call")
    print(bs.calc_bsprice(S,K,r,y,sigma,T, "call"))
    print(np.mean(c))










