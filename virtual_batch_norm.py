class VirtualBatchNorm:

    def __init__(self):

        """Batch norm normalizes the activations of each units and adds
        learnable parameters gamma and beta to scale them back.
        Bias can be left out because of beta.

        In VBN, we collect a reference batch every X steps and simply use those params"""

        self.means = dict()
        self.stds = dict()

    def update_stats(self, mean, std, key):

        self.means[key] = mean
        self.stds[key] = std

    def normalize_activations(self, activations, key):

        activations -= self.means[key]
        activations /= self.stds[key]
        return activations

    def denormalize_activations(self, activations, key):

        """Just transform back using the running statistics."""
        activations *= self.stds[key]
        activations += self.means[key]
        return activations



