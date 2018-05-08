import numpy as np

class AdamOptimizer:

    def __init__(self, param, beta1=0.9, beta2=0.999, epsilon=1e-08):

        self.dim = param.shape
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.momentum = np.zeros(self.dim, dtype=np.float32)
        self.rms = np.zeros(self.dim, dtype=np.float32)


    def compute_update(self, gradient):

        #self.t += 1

        #self.momentum = self.beta1 * self.momentum + (1 - self.beta1) * gradient
        #self.momentum /= (1-self.beta1 ** self.t)

        #self.rms = self.beta2 * self.rms + (1 - self.beta2) * (gradient * gradient)
        #self.rms /= (1 - self.beta2 ** self.t)

        #step = self.momentum #/ (np.sqrt(self.rms) + self.epsilon)

        #print(np.mean(np.abs(step)))

        return gradient#step

# vector = np.random.normal(0, 1, [1, 4])
# adam = AdamOptimizer(param=vector)
#
#
# for i in range(100):
#     vector = np.random.normal(0,.1, [1,4])
#     #vector[0, 0] = 1
#     print('Vector')
#     print(vector)
#     update = adam.compute_update(vector)
#     print('Update')
#     print(update)
#     print(np.mean(np.abs(update)))