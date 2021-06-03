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
        self.activation = lambda x: max(0, min(1, x))
        self.activation_grad = lambda x: 1 if 0 <= x <= 1 else 0
        

    def init_weights(self, shape: np.ndarray):
        """
        initialize weights acc to Glorot initialization
        """
        weight_shape = tuple([(shape[i + 1], shape[i] + 1) for i in range(len(shape) - 1)])

        # TODO: initialize using Glorot

        return [np.zeros(weight_shape[i]) for i in range(len(weight_shape))]

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
        
        # TODO: configure to apply element wise

        activations = [x] + [self.activation(self.state[i]) for i in range(len(self.state))]

        # TODO: compute product of activations with weights (refer to original code)

        return magnitudes

    def energy_grad_state(self, x):
        """
        returns the gradient of energy function evaluated at current state
        """
        # TODO: work out by hand (refer to original code)

        return 0

    def energy_grad_weights(self, x):
        """
        returns the gradient of energy function evaluated at current state
        """
        # TODO: work out by hand (refer to original code)

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
        self.activation = lambda x: max(0, min(1, x)) + 0.01 * x
        self.activation = lambda x: 1 if 0 <= x <= 1 else 0.01
    
    def evaluate(self):
        """
        returns output of the net
        """
        return 0
