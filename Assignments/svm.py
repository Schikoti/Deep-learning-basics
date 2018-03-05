import numpy as np


class SVM(object):
    def __init__(self, n_epochs=10, lr=0.1, l2_reg=1):
        """
        """
        self.b = None
        self.w = None
        self.n_epochs = n_epochs
        self.lr = lr
        self.l2_reg = l2_reg

    def forward(self, x):
        """
        Compute "forward" computation of SVM f(x) = w^T + b

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features

        Returns
        -------
        np.array
            A 1 dimensional vector of the logistic function
        """
        f = (np.dot(self.w, np.transpose(x)) + self.b)
        return f

    def loss(self, x, y):
        """
        Return the SVM hinge loss
        L(x) = max(0, 1-y(f(x))) + 0.5 w^T w

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels

        Returns
        -------
        float
            The logistic loss value
        """
        loss = np.mean(np.maximum(0, (1-(y * (np.dot(self.w, np.transpose(x)) + self.b))))) + (self.l2_reg*(np.dot(self.w, np.transpose(self.w))))/2
        return loss[0,0]

    def grad_loss_wrt_b(self, x, y):
        """
        Compute the gradient of the loss with respect to b

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels

        Returns
        -------
        float
            The gradient
        """
        dlb=[]
        lb=y*self.forward(x)
        for i in range(0, lb.shape[1]):
            if lb[0][i]<1:
                dlb.append(-y[i])
            elif lb[0][i]>1:
                dlb.append(0)
        dlb = ((np.sum(dlb))*1.0) / (len(dlb))
        return dlb



    def grad_loss_wrt_w(self, x, y):
        """
        Compute the gradient of the loss with respect to w

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels

        Returns
        -------
        np.array
            The gradient (should be the same size as self.w)
        """
        dlw=[]
        lw = (y * (np.dot(self.w, np.transpose(x)) + self.b))
        for i in range(0, lw.shape[1]):
            if lw[0][i] < 1:
                dlw.append((-y[i]*x[i]) + self.l2_reg*self.w)
            elif lw[0][i] > 1:
                dlw.append(self.l2_reg*self.w)
        return np.asarray(np.mean(dlw,axis=0))

    def fit(self, x, y, plot=False):
        """
        Fit the model to the data

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels
        """
        self.w = np.random.rand(1, x.shape[1])
        self.b = 0
        lossf = []
        for i in range(1, self.n_epochs + 1):
            temp = self.b - self.lr * self.grad_loss_wrt_b(x, y)
            self.w = self.w - self.lr * self.grad_loss_wrt_w(x, y)
            self.b = temp
            lf = self.loss(x, y)
            lossf.append(lf)
        return lossf

    def predict(self, x):
        """
        Predict the labels for test data x

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features

        Returns
        -------
        np.array
            Vector of predicted class labels for every training sample
        """
        yfinal = []
        ytest = self.forward(x)
        for i in ytest[0]:
            if i < 0:
                y = -1
            elif i >= 0:
                y = 1
            yfinal.append(y)
        return np.asarray(yfinal)
