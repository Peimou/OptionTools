

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