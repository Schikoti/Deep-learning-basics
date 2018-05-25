import numpy as np


class MaxPoolLayer(object):
    def __init__(self, size=2):
        """
        MaxPool layer
        Ok to assume non-overlapping regions
        """
        self.locs = None  # to store max locations
        self.size = size  # size of the pooling

    def forward(self, x):
        """
        Compute "forward" computation of max pooling layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of input channels x number of rows x number of columns

        Returns
        -------
        np.array
            The output of the maxpooling

        Stores
        -------
        self.locs : np.array
             The locations of the maxes (needed for back propagation)
        """

        max_final=np.zeros(x.shape)
        max_repeated=np.zeros(x.shape)
        self.x=x
        self.locs=np.zeros((x.shape))
        max_vals=np.zeros((x.shape[0],x.shape[1],x.shape[2]/self.size,x.shape[3]/self.size))
        rem1=x.shape[2]%(self.size)
        rem2 = x.shape[3]%(self.size)
        x_new=x[:,:,0:(x.shape[2]-rem1),0:(x.shape[3]-rem2)]
        for s in range(x.shape[0]):
            for c in range(x.shape[1]):
                for i in range(0,x_new.shape[2],self.size):
                    for j in range(0,x_new.shape[3], self.size):
                        x_sliced = x[s,c,i:i+self.size,j:j+self.size]
                        max_vals[s,c,i/self.size,j/self.size] = np.max(x_sliced)

        max_final = np.repeat(np.repeat(max_vals,self.size,axis=2),self.size,axis=3)
        self.locs[:,:,0:x_new.shape[2],0:x_new.shape[3]] = (max_final==x_new)
        return max_vals


    def backward(self, y_grad):
        """
        Compute "backward" computation of maxpool layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input
        """
        y_grad_new=np.repeat(np.repeat(y_grad,self.size,axis=2),self.size,axis=3)
        x_grad=np.zeros(self.locs.shape)
        x_grad[:,:,0:y_grad_new.shape[2],0:y_grad_new.shape[3]]=np.multiply(self.locs[:,:,0:y_grad_new.shape[2],0:y_grad_new.shape[3]],y_grad_new)
        return x_grad

    def update_param(self, lr):
        pass
