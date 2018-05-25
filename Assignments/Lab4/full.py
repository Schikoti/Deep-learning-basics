import numpy as np


class FullLayer(object):
    def __init__(self, n_i, n_o):
        """
        Fully connected layer

        Parameters
        ----------
        n_i : integer
            The number of inputs
        n_o : integer
            The number of outputs
        """

        # need to initialize self.W and self.b
        self.n_i = n_i
        self.n_o = n_o
        self.x = None
        self.W_grad = None
        self.b_grad = None
        w_sd = np.sqrt(2.0 / (n_i + n_o)).astype(float)
        self.W = np.random.normal(0.0, w_sd, (n_o, n_i)).astype(float)
        self.b = np.zeros(([1, n_o]), 'float64')


    def forward(self, x):
        """
        Compute "forward" computation of fully connected layer

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
        self.x : np.array
             The input data (need to store for backwards pass)
        """
        self.x = x
        f_full2 = np.dot(self.x, np.transpose(self.W)) + self.b
        return f_full2

    def backward(self, y_grad):
        """
        Compute "backward" computation of fully connected layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input

        Stores
        -------
        self.b_grad : np.array
             The gradient with respect to b (same dimensions as self.b)
        self.W_grad : np.array
             The gradient with respect to W (same dimensions as self.W
        """
        dlx = np.dot(y_grad, self.W)
        self.W_grad = np.zeros((self.n_o, self.n_i))
        for i in range(0, self.x.shape[0]):
            self.W_grad += np.dot((y_grad[i].reshape((1, self.n_o))).T, self.x[i].reshape((1, self.n_i)))
        self.b_grad = (np.sum(y_grad, axis=0).reshape(1, y_grad.shape[1]))
        return dlx

    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate

        Stores
        -------
        self.W : np.array
             The updated value for self.W
        self.b : np.array
             The updated value for self.b
        """
        self.b = self.b - lr * self.b_grad
        self.W = self.W - lr * self.W_grad
