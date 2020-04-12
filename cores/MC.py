import numpy as np
import matplotlib.pyplot as plt
from cores import Random

class MC_Path():
    def __init__(self, s0, mu, sigma, T, N_t, N_path, path, **config):
        self.Path = path
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.N_t = N_t
        self.N_path = N_path
        self.s0 = s0

        for item in config.items():
            if hasattr(self, item[0]):
                setattr(self, item[0], item[1])

class ArrayBT(object):
    def __init__(self, s0, r, y, T, Nt):
        self.s0 = s0
        self.r = r
        self.y = y
        self.T = T
        self.Nt = Nt
        self.dt = T/Nt
        self.sArray = np.zeros((Nt + 1,Nt + 1))
        self.sArray[0][0] = s0


class MarkovChain(object):
    def __init__(self, num_state, T, mctype = "D", **kwargs):
        self.num_state = num_state
        self.T = T
        self.mctype = (mctype.lower() == "d")
        self.N_t = -1

        if self.mctype:
            for item in kwargs.items():
                if hasattr(self, item[0]):
                    setattr(self, item[0], item[1])

            if self.N_t < 0:
                raise ValueError("Need valid steps of simulation, "
                                 "reset N_t in construct function")

    def __repr__(self):
        type = "Discrete" if self.mctype else "Continuous"
        return f"{type} Markov Chain  T: {self.T}, State Number: {self.num_state}"


    @property
    def Pmatrix(self):
        return self.__Pmatrix


    @Pmatrix.setter
    def Pmatrix(self, matrix):
        if not self.mctype:
            raise ValueError("P matrix is a property of Discrete Markov Chain")
        if len(matrix.shape) == 2:
            if not np.allclose(np.sum(matrix, axis = 1), np.ones(self.num_state)):
                raise ValueError("In Invalid Transition matrix. The sum of rows must be 1.")
            self.__Pmatrix = np.tile(matrix, (self.N_t, 1, 1))
            self.TimeHomogenous = True
        elif len(matrix.shape) == 3 and matrix.shape[0] == self.N_t:
            if not np.allclose(np.sum(matrix, axis = 1), np.ones((self.N_t, self.num_state))):
                raise ValueError("In Invalid Transition matrix. The sum of rows must be 1.")
            self.TimeHomogenous = False
            self.__Pmatrix = matrix
        else:
            raise ValueError("Invalid Transition matrix.")


    @property
    def Amatrix(self):
        return self.__Amatrix


    @Amatrix.setter
    def Amatrix(self, matrix:np.array):
        '''
        matrix can be 2d function matrix
        '''
        if self.mctype:
            raise ValueError("A matrix is a property of Continuous Markov Chain")

        if len(matrix.shape) == 2:
            self.checkAmatrix(matrix, self.num_state)
            self.TimeHomogenous = True
            self.__Amatrix = matrix
        elif hasattr(matrix[0][0], '__call__'):
            self.TimeHomogenous = False
            self.__Amatrix = matrix
        else:
            raise ValueError("Invalid Transition matrix.")


    @staticmethod
    def checkAmatrix(matrix, num_state):
        if not np.all(np.diag(matrix) <= 0):
            raise ValueError("The diagonal of matrix must be negative")
        if not np.allclose(np.sum(matrix, axis=1), np.zeros(num_state)):
            raise ValueError("Invalid Transition matrix. The sum of rows must be 0.")


    def csimulate(self, init_state, rst = None, rss = None,
                  verbose = True):

        if not self.TimeHomogenous:
            raise ValueError("Only numerical matrix is available in this version.")

        state = np.arange(self.num_state, dtype=int)
        lmblist = -np.diag(self.Amatrix)
        nsize = max(int(self.T / Random.expoDist(np.max(lmblist)).mean()),1)

        self.path = np.empty(2 * nsize, dtype=int)
        self.tpath = np.empty(2 * nsize, dtype=float)

        if init_state in state:
            self.path[0] = init_state
            self.tpath[0] = 0
        else:
            raise ValueError("Invalid initial state.")

        if verbose:
            print(f'Iter 0: initial state {self.path[0]}')
        if rst != None:
            np.random.seed(rst)
        ut = np.random.rand(2 * nsize)
        if rss != None:
            np.random.seed(rss)
        us = np.random.rand(2 * nsize)
        index, t = 1, 0
        while(t <= self.T):
            if index >= len(self.path): #This should happen rarely
                self.path = np.r_[self.path, np.empty(nsize, dtype=int)]
                self.tpath = np.r_[self.tpath, np.empty(nsize, dtype=float)]
                if rst != None:
                    np.random.seed(rst)
                ut = np.random.rand(2 * nsize)
                if rss != None:
                    np.random.seed(rss)
                us = np.random.rand(2 * nsize)

            cur = self.path[index - 1]
            self.tpath[index] = -1/lmblist[cur] * np.log(ut[index - 1])
            P = -self.Amatrix[cur, :] / self.Amatrix[cur, cur]
            cumP = np.cumsum(np.where(P>0, P, 0))
            self.path[index] = np.min(state[us[index - 1] <= cumP])
            t += self.tpath[index]
            if verbose:
                print(f'Iter {index}: {self.path[index - 1]} --> {self.path[index]} '
                      f'in {self.tpath[index]}')
            index += 1

        self.tpath = np.cumsum(self.tpath[:index-1])
        self.path = self.path[:index-1]
        return self.path, self.tpath


    def dsimulate(self, init_state:int, random_seed = None, verbose = True):
        self.tpath = np.linspace(0, self.T, self.N_t + 1)
        self.path = np.empty(self.N_t + 1, dtype=int)
        state = np.arange(self.num_state, dtype=int)
        if init_state in state:
            self.path[0] = init_state
        else:
            raise ValueError("Invalid initial state.")
        if verbose:
            print(f'Iter 0: initial state {self.path[0]}')
        if random_seed != None:
            np.random.seed(random_seed)
        u = np.random.rand(self.N_t)

        for i in range(1, self.N_t + 1):
            cur = self.path[i - 1]
            cumP = np.cumsum(self.Pmatrix[i - 1], axis = 1)
            p = cumP[cur, :]
            self.path[i] = np.min(state[u[i - 1] <= p])
            if verbose:
                print(f'Iter {i}: {self.path[i - 1]} --> {self.path[i]}')
        return self.path, self.tpath


    def simulate(self, init_sate, rst = None, rss = None, verbose = True):
        if self.mctype:
            return self.dsimulate(init_sate, rss, verbose)
        else:
            return self.csimulate(init_sate, rst, rss, verbose)


    def show(self, figsize = (8,8), annotate = False):
        if not hasattr(self, "path"):
            raise ValueError("No available simulations to plot. ")
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.plot(self.tpath, self.path, "o--")

        if annotate:
            for i in range(len(self.path)):
                ax.annotate(self.path[i],
                            (self.tpath[i], self.path[i]),
                            (self.tpath[i], 0.95 * self.path[i]))
        plt.xlabel('iteration')
        plt.ylabel('state')
        plt.title(self.__repr__())
        plt.show()



if __name__ == "__main__":
    mc = MarkovChain(num_state=5, T=15, mctype="C")
    am = np.random.uniform(0,1,25).reshape(5,5)
    for i in range(len(am)):
        am[i][i] -= np.sum(am[i,:])
    mc.Amatrix = am
    mc.simulate(1, verbose=False)
    mc.show((10,4))

    mc = MarkovChain(num_state=10, T=1, N_t = 50, mctype="D")
    pm = np.random.uniform(0, 1, 100).reshape(10, 10)
    sum_rows = np.sum(pm, axis=1)
    P = (pm.T / sum_rows).T  # rows should add up to 1
    mc.Pmatrix = P
    mc.simulate(1, verbose=False)
    mc.show((10, 4))
5