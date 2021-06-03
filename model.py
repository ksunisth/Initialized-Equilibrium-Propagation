"""
defines a model object
"""

import numpy as np

class Equilibrium:
    """
    train nn using equilibrium propagation
    """

    def __init__(self, shape: np.ndarray):
        """
        shape = [d1, d2, ..., dn]
        di is dimentionality of the ith layer in a FCN
        """
        self.input_dim = shape[0]
        self.state = [np.zeros(shape[i]) for i in range(1, len(shape))]
        self.weights = self.init_weights(shape)

    @staticmethod
    def rho(v):
        """
        activation function to be used
        """
        t = np.clip(v, 0, 1)
        return t

    @staticmethod
    def rhophrime(v):
        """
        gradient of activation function
        """
        v = np.asarray(v)
        return ((v >= 0) & (v <= 1)).astype(int)

    @staticmethod
    def init_weights(shape: np.ndarray):
        """
        initialize weights acc to Glorot initialization
        """
        def get_initialized_layer(n_in, n_out):
            """
            perform Glorot initialization of a single layer of the net
            input dim: n_in
            output dim: n_out
            """
            rng = np.random.RandomState()
            return np.asarray(np.random.uniform(-np.sqrt(6 / (n_in + n_out)), np.sqrt(6 / (n_in + n_out)), (n_in, n_out)))

        weight_shape = zip(shape[:-1], shape[1:])
        return [get_initialized_layer(n_in, n_out) for n_in, n_out in weight_shape]

    def outputs(self):
        """
        returns output of the net
        """
        return self.state[-1]

    def negative_phase(self, x, num_steps: int, step_size:float):
        """
        neg phase training
        """
        for _ in range(num_steps):
            self.state -= step_size * self.energy_grad_state(x)

    def positive_phase(self, x, y, num_steps: int, step_size: float, beta: float):
        """
        pos phase training
        """
        for _ in range(num_steps):
            self.state -= step_size + self.clamped_energy_grad(x, y, beta)

    def energy(self, x):
        """
        returns energy of the net
        """
        magnitudes = sum([np.sum(state ** 2) for state in self.state]) / 2

        activations = rho(self.state)

        # TODO: #5 compute product of activations with weights (refer to original code)

        return magnitudes

    def energy_grad_state(self, x):
        """
        returns the gradient of energy function evaluated at current state
        """
        # TODO: #6 work out by hand (refer to original code)

        return 0

    def energy_grad_weights(self, x):
        """
        returns the gradient of energy function evaluated at current state
        """
        # TODO: #7 work out by hand (refer to original code)

        return 0

    def clamped_energy_grad(self, x, y, beta):
        """
        returns gradient of clamped energy function at current state and target
        """
        return self.energy_grad_state(x) + 2 * beta * (y - self.outputs())


class Initialize:
    """
    a feedforward nn trained to initialize state of equilibrium net
    """

    def __init__(self):
        pass
    
    def evaluate(self):
        """
        returns output of the net
        """
        return 0

if __name__ == "__main":
    import doctest
    doctest.testmod
