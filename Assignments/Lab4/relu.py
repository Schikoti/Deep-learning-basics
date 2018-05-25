import numpy as np

class ReluLayer(object):
    def __init__(self):
        """
        Rectified Linear Unit
        """
        self.y = None

    def forward(self, x):

        """
        Implement forward pass of Relu

        y = x if x > 0
        y = 0 otherwise

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
             The output data (need to store for backwards pass)
        """

        self.y = np.zeros(x.shape)
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                for k in range(0, x.shape[2]):
                    for l in range(0, x.shape[3]):
                        if x[i][j][k][l] > 0:
                            self.y[i][j][k][l] = x[i][j][k][l]
                        else:
                            self.y[i][j][k][l] = 0
        return self.y

    def backward(self, y_grad):
        """
        Implement backward pass of Relu

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input
        """
        x = self.y
        relu_backward = np.zeros(x.shape)
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1]):
                for k in range(0, x.shape[2]):
                    for l in range(0, x.shape[3]):
                        if x[i][j][k][l] > 0:
                            relu_backward[i][j][k][l] = 1
                        else:
                            relu_backward[i][j][k][l] = 0
        return relu_backward * y_grad

    def update_param(self, lr):
        pass  # no parameters to update
