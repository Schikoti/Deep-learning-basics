import numpy as np


class SoftMaxLayer(object):
    def __init__(self):
        """
        Constructor
        """
        self.y = None

    def forward(self, x):
        """
        Implement forward pass of softmax

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features

        Returns
        -------
        np.array
            The output of the layer

        Stores
        -------
        self.y : np.array
             The output of the layer (needed for backpropagation)
        """
        x_new = x - np.max(x, axis=1).reshape((x.shape[0], 1))
        fsm = np.exp(x_new) / (np.sum(np.exp(x_new), axis=1)).reshape((x_new.shape[0], 1))
        self.y = fsm
        return fsm

    def backward(self, y_grad):
        """
        Compute "backward" computation of softmax

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input

        """
        z = self.y
        dly = np.zeros([y_grad.shape[0], y_grad.shape[1]])
        for i in range(0, z.shape[0]):
            d = np.diag(z[i])
            c = np.outer(z[i].T, z[i])
            dzy = d - c
            dly[i] = np.dot(y_grad[i], dzy)

        return dly

    def update_param(self, lr):
        pass  # no learning for softmax layer
