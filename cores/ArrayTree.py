import numpy as np

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







