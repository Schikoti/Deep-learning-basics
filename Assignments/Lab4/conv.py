import numpy as np
import scipy.signal


class ConvLayer(object):
    def __init__(self, n_i, n_o, h):
        """
        Convolutional layer

        Parameters
        ----------
        n_i : integer
            The number of input channels
        n_o : integer
            The number of output channels
        h : integer
            The size of the filter
        """
        # glorot initialization


        self.n_i = n_i
        self.n_o = n_o
        self.h=h
        w_sd = np.sqrt(2.0 / (n_i + n_o)).astype(float)
        self.W = np.random.normal(0.0, w_sd, (n_o, n_i, h, h)).astype(float)
        self.b=np.zeros((1,n_o))
        self.W_grad = None
        self.b_grad = None
        self.x_grad = None

    def forward(self, x):
        """
        Compute "forward" computation of convolutional layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of input channels x number of rows x number of columns

        Returns
        -------
        np.array
            The output of the convolutiona

        Stores
        -------
        self.x : np.array
             The input data (need to store for backwards pass)
        """
        out_final=np.zeros((x.shape[0],self.n_o,x.shape[2],x.shape[3]))
        out_forward=np.zeros((x.shape[0],self.n_o,x.shape[2],x.shape[3]))
        self.x = x

        for i in range(x.shape[0]):
            for j in range(self.n_o):
                for m in range(self.n_i):
                    img=self.x[i,m,:,:]
                    filter=self.W[j,m,:,:]
                    out_forward[i,j,:,:]+=scipy.signal.correlate(img, filter, mode='same')
                b_o = np.full((x.shape[2], x.shape[3]), self.b[0, j])
                out_forward[i,j]+=b_o
        return(out_forward)


    def backward(self, y_grad):
        """
        Compute "backward" computation of convolutional layer

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
        self.w_grad : np.array
             The gradient with respect to W (same dimensions as self.W
        """

        out_xgrad = np.zeros((y_grad.shape))
        self.b_grad= np.sum(y_grad, axis=(0,2,3)).reshape((1,self.n_o))


        n_b=self.x.shape[0]
        self.W_grad = np.zeros(self.W.shape)
        for i in range(n_b):
            for j in range(self.n_o):
                for m in range(self.n_i):
                    p=(self.h-1)/2
                    x_padded=np.pad(self.x[i,m,:,:],p,'constant')
                    self.W_grad[j,m,:,:]+=scipy.signal.correlate(x_padded, y_grad[i,j,:,:],mode='valid')

        out_xgrad=np.zeros(self.x.shape)
        for i in range(n_b):
            for j in range(self.n_o):
                for m in range(self.n_i):
                    y_img=y_grad[i,j,:,:]
                    filter = self.W[j, m,:,:]
                    out_xgrad[i, m,:,:] += scipy.signal.convolve(y_img, filter, mode='same')
        return(out_xgrad)

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
        self.W=self.W-(lr*self.W_grad)
        self.b = self.b - (lr * self.b_grad)
