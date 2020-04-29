import numpy as np
from math import erf, pi
from abc import abstractmethod

class Distribution(object):

    @abstractmethod
    def pdf(self, **config):
        pass


    def cdf(self, **config):
        pass


    def mean(self):
        pass


    def var(self):
        pass

    @abstractmethod
    def simulate(self, **config):
        pass


    def MGF(self, **config):
        pass
    


class Gaussian(Distribution):
    '''
    mu: the mean of Gaussain
    sigma: standard deviation
    '''
    def __init__(self, mu, sigma, random_seed = None):
        self.mu = mu
        self.sigma = sigma
        self.random_seed = random_seed

    def pdf(self, x):
        return np.exp(- np.power(x - self.mu, 2) / (2 * self.sigma**2))\
                         / (np.sqrt(2 * pi) * self.sigma)


    def cdf(self, x):
        return 0.5 * (1.0 + erf((x-self.mu) / (np.sqrt(2.0) * self.sigma)))


    def mean(self):
        return self.mu


    def var(self):
        return self.sigma**2


    def simulate(self, size = 1):
        # Just copy numpy for better coupling.
        return np.random.normal(self.mu, self.sigma, size=size)


    def MGF(self, t):
        return np.exp(self.mu * t + 0.5 * np.power(self.sigma * t, 2))


    def muSFunc(self, x):
        '''
        Score function of Gaussian to mu.

        Reference:
        ----------
        https://en.wikipedia.org/wiki/Score_function
        '''
        return (x - self.mu) / np.power(self.sigma,2)


    def sigmaSFunc(self, x):
        '''
        Score function of Guassian to sigma
        '''
        return -self.pdf(x)/self.sigma + self.pdf(x) * (x - self.mu)**2/np.power(self.sigma, 3)


    def __repr__(self):
        return "Gaussian Distribution"



class expoDist(Distribution):
    def __init__(self, lmb, random_seed = None):
        self.lmb = lmb
        self.random_seed = random_seed

    def pdf(self, x):
        return self.lmb * np.exp(-self.lmb * x)


    def cdf(self, x):
        return 1 - np.exp(-self.lmb * x)


    def mean(self):
        return 1/self.lmb


    def var(self):
        return 1/self.lmb**2


    def simulate(self, size = 1):
        if self.random_seed != None:
            np.random.seed(self.random_seed)
        return -1/self.lmb * np.log(np.random.uniform(0,1,size))

    def MGF(self, t):
        return self.lmb / (self.lmb - t)

    def __repr__(self):
        return "Exponential Distribution"


class MultiUniform(Distribution):
    def __init__(self, mu: np.array, eta: np.float64, lower: np.float64, upper: np.float64):
        self.mu = np.asarray(mu)
        self.lower = lower
        self.upper = upper
        self.eta = eta

    def pdf(self, x):
        indi = np.all(x - self.mu < self.upper) and np.all(x - self.mu >= self.lower)
        return np.power(1 / (self.upper - self.lower), len(self.mu)) * int(indi)

    def simulate(self, size=1):
        return self.mu + self.eta * np.random.uniform(self.lower, self.upper, size=(size, *self.mu.shape))


class MultiGaussian(Distribution):
    def __init__(self, mu: np.array, sigma: np.array):
        self.mu = np.asarray(mu)
        self.sigma = np.asarray(sigma)
        if len(mu) != len(sigma):
            raise ValueError("The sizes of mu and sigma are different")

    def pdf(self, x):
        n = len(self.mu)
        mu = self.mu.reshape(-1,1)
        x = x.reshape(-1,1)
        coef = 1 / np.power(2 * np.pi, n / 2) / np.sqrt(np.linalg.det(self.sigma))
        return coef * np.exp(-0.5*np.squeeze((x - mu).T @ np.linalg.pinv(self.sigma) @ (x - mu)))


    def simulate(self, size=1):
        return np.random.multivariate_normal(self.mu, self.sigma, size = size)


if __name__ == "__main__":
    a = np.linspace(-10,10,500)