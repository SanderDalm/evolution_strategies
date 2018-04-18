class VirtualBatchNorm:

    def __init__(self, params):

        """Batch norm normalizes the activations of each units and adds
        learnable parameters gamma and beta to scale them back.
        Bias can be left out because of beta.

        In VBN, we collect a reference batch every X steps and simply use those params"""


        self.running_avg = dict()
        self.running_std = dict()

        for key in params.keys():
            self.running_avg[key] = 0
            self.running_std[key] = 0

    def update_stats(self, activations):

        # .9 * self.running_avg[key] + .1 * activations
        pass

    def normalize_activations(self, activations, key):

        activations -= self.running_avg[key]
        activations /= self.running_std[key]
        return activations

    def denormalize_activations(self, activations, key):

        """Just transform back using the running statistics."""
        activations *= self.running_std[key]
        activations += self.running_avg[key]
        return activations



