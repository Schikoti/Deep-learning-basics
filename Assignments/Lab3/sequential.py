from __future__ import print_function
import numpy as np


class Sequential(object):
    def __init__(self, layers, loss):
        """
        Sequential model

        Implements a sequence of layers

        Parameters
        ----------
        layers : list of layer objects
        loss : loss object
        """
        self.layers = layers
        self.loss = loss

    def forward(self, x, target=None):
        """
        Forward pass through all layers
        
        if target is not none, then also do loss layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features
        target : np.array
            The target data of size number of training samples x number of features (one-hot)

        Returns
        -------
        np.array
            The output of the model
        """
        for i in range(0, len(self.layers)):
            x = self.layers[i].forward(x)
        if target is None:
            forwardout = x
            return forwardout
        else:
            forwardout = self.loss.forward(x, target)
            return forwardout

    def backward(self):
        """
        Compute "backward" computation of fully connected layer

        Returns
        -------
        np.array
            The gradient at the input

        """
        y_grad = self.loss.backward()
        layers_len = len(self.layers) - 1
        for i in range(0, len(self.layers)):
            y_grad = self.layers[layers_len - i].backward(y_grad)
        return y_grad

    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate
        """
        for i in range(len(self.layers)):
            self.layers[i].update_param(lr)

    def fit(self, x, y, epochs=10, lr=0.1, batch_size=128):
        """
        Fit parameters of all layers using batches

        Parameters
        ----------
        x : numpy matrix
            Training data (number of samples x number of features)
        y : numpy matrix
            Training labels (number of samples x number of features) (one-hot)
        epochs: integer
            Number of epochs to run (1 epoch = 1 pass through entire data)
        lr: float
            Learning rate
        batch_size: integer
            Number of data samples per batch of gradient descent
        """
        finalloss = []
        if x.shape[0] % batch_size is 0:
            iterations = x.shape[0] / batch_size
        else:
            iterations = (x.shape[0] / batch_size) + 1
        for e in range(epochs):
            b = 0
            loss = []
            for i in range(iterations):
                if i == (iterations) - 1:
                    x1 = x[b:, :]
                    y1 = y[b:, :]

                else:
                    x1 = x[b:b + batch_size, :]
                    y1 = y[b:b + batch_size, :]

                loss.append(self.forward(x1, y1))
                backward_out = self.backward()
                self.update_param(lr)
                b += batch_size
            print('epochs ', e, 'iterations ', i, 'loss ', loss[-1])
            finalloss.append(np.mean(loss))
        return finalloss

    def predict(self, x):
        """
        Return class prediction with input x

        Parameters
        ----------
        x : numpy matrix
            Testing data data (number of samples x number of features)

        Returns
        -------
        np.array
            The output of the model (integer class predictions)
        """
        prob = self.forward(x)
        ylabel_pred = np.argmax(prob, axis=1)
        return ylabel_pred
