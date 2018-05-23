import numpy as np

class Optimizer:

    def __init__(self, param, beta1=0.9, beta2=0.9, epsilon=1e-08, adam=False):

        self.dim = param.shape
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.momentum = np.zeros(self.dim, dtype=np.float32)
        self.rms = np.zeros(self.dim, dtype=np.float32)
        self.adam = adam


    def compute_update(self, gradient):

        self.t += 1

        self.momentum = self.beta1 * self.momentum + (1 - self.beta1) * gradient

        if self.adam:
            self.rms = self.beta2 * self.rms + (1 - self.beta2) * (gradient * gradient)
            step = self.momentum / (np.sqrt(self.rms) + self.epsilon)
            return step
        else:
            step = self.momentum
            return step