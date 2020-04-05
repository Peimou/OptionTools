import numpy as np
import numba as nb
import time


def iteration(x, f1, f2, tol, itermax):
    ntol = np.abs(f1(x)/f2(x))
    for i in range(itermax):
        if ntol < tol:
            return x
        ntol = np.abs(f1(x) / f2(x))
        x = x - f1(x)/f2(x)

    return x



class Newton:
    def __init__(self, x, f1, f2, tol = 1e-5, itermax = 10000):
        self.x = x
        self._f1 = nb.njit()(f1)
        self._f2 = nb.njit()(f2)
        self._tol = tol
        self._itermax = itermax


    def iteration(self):
        return nb.njit()(iteration)(self.x, self._f1, self._f2, self._tol, self._itermax)




if __name__ == "__main__":
    #examples
    f1 = lambda x: x * x - 2*x
    f2 = lambda x: 2*x - 2
    x = 1.5
    t = time.time()
    y = iteration(x, f1, f2, 1e-5, 10000)
    t = time.time() - t
    print(y)
    print(t)
