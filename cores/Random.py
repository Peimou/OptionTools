import numpy as np
from abc import abstractmethod

class Distribution(object):

    @abstractmethod
    def pdf(self):
        pass

    @abstractmethod
    def cdf(self):
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


class expoDist(Distribution):
    def __init__(self, lmb, random_seed = None):
        self.lmb = lmb
        self.random_seed = random_seed

    def pdf(self):
        return lambda x: self.lmb * np.exp(-self.lmb * x)


    def cdf(self):
        return lambda x: 1 - np.exp(-self.lmb * x)


    def mean(self):
        return 1/self.lmb


    def var(self):
        return 1/self.lmb**2


    def simulate(self, size = 1):
        if self.random_seed != None:
            np.random.seed(self.random_seed)
        return -1/self.lmb * np.log(np.random.uniform(0,1,size))

    def __repr__(self):
        return repr("Exponential Distribution")




if __name__ == "__main__":
    exp = expoDist(0.1)
    a = exp.simulate(10000)
    print(np.mean(a))
    print(exp.mean())
