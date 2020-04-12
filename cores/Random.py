import numpy as np
from math import erf, pi
from abc import abstractmethod

class Distribution(object):

    @abstractmethod
    def pdf(self, **config):
        pass

    @abstractmethod
    def cdf(self, **config):
        pass

    @abstractmethod
    def mean(self):
        pass

    @abstractmethod
    def var(self):
        pass

    @abstractmethod
    def simulate(self, **config):
        pass

    @abstractmethod
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




if __name__ == "__main__":
    a = np.linspace(-10,10,500)
    G
