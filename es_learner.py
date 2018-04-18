import numpy as np
from optimizers import Adam
from virtual_batch_norm import VirtualBatchNorm
class ESLearner:

    def __init__(self,
                 input_dims,
                 output_dims,
                 hidden_size,
                 output_lower_bound,
                 output_upper_bound,
                 sigma,
                 alpha,
                 pop,
                 env,
                 discrete):

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.output_lower_bound = output_lower_bound
        self.upper_lower_bound = output_upper_bound
        self.sigma = sigma
        self.alpha = alpha
        self.pop = pop
        self.env = env
        self.discrete = discrete
        self.params =   {
                        'w1': np.random.normal(0, .5, [input_dims, hidden_size]),
                        'b1': np.zeros([hidden_size, 1]),
                        'w2': np.random.normal(0, .5, [hidden_size, hidden_size]),
                        'b2': np.zeros([hidden_size, 1]),
                        'w3': np.random.normal(0, .5, [hidden_size, output_dims]),
                        'b3': np.zeros([output_dims, 1])
                        }

        #self.optimizers = dict()
        #for key in self.params.keys():
        #    self.optimizers[key] = Adam()

        #self.VBN = VirtualBatchNorm(params)

    def load_params(self, params):
        self.params = params

    def generate_noise(self, x):
        noise = np.random.normal(0, self.sigma, [x.shape[0], x.shape[1]])
        return noise


    def activation(self, x):
        return np.maximum(0, x)


    def model(self, x, params):

        a1 = self.activation(np.matmul(params['w1'].T, x) + params['b1'])
        a2 = self.activation(np.matmul(params['w2'].T, a1) + params['b2'])
        if self.discrete:
            return np.argmax(self.activation(np.matmul(params['w3'].T, a2) + params['b3']))
        else:
            return np.clip(np.matmul(params['w3'].T, a2) + params['b3'], self.output_lower_bound, self.upper_lower_bound)


    def run_episode(self, params):

        episode_reward = 0
        done = False
        observation = self.env.reset()
        x = observation.reshape([self.input_dims, 1])

        while not done:
            action = self.model(x, params)
            observation, reward, done, info = self.env.step(action)
            x = observation.reshape([self.input_dims, 1])
            episode_reward += reward

        return episode_reward


    def compute_ranks(self, x):
        """
        Returns ranks in [0, len(x))
        Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
        """
        assert x.ndim == 1
        ranks = np.empty(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        return ranks


    def compute_centered_ranks(self, x):
        y = self.compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
        y /= (x.size - 1)
        y -= .5
        return y


    def run_generation(self):

        episode_params = dict()
        noises = dict()
        for key in self.params.keys():
            noises[key] = []
        rewards = []

        for _ in range(self.pop):

            for key in self.params.keys():
                noise = self.generate_noise(self.params[key])
                episode_params[key] = self.params[key] + noise
                noises[key].append(noise)

            rewards.append(self.run_episode(episode_params))

        ranking = self.compute_centered_ranks(np.array(rewards))

        for key in self.params.keys():
            update = self.compute_update(noises[key], ranking)
            self.update_params(update, key)

        return np.mean(rewards), self.params


    def compute_update(self, noises, ranking):

        noises = np.array(noises)
        update = np.zeros([noises.shape[1], noises.shape[2]])

        for index, r in enumerate(ranking):
            update += noises[index] * r

        return update


    def update_params(self, update, key):

        self.params[key] += update * (self.alpha/(self.pop*self.sigma))