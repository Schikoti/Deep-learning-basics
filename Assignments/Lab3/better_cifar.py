import numpy as np
import matplotlib.pyplot as plt

from layers import (FullLayer, ReluLayer, SoftMaxLayer, CrossEntropyLayer)
from layers.sequential import Sequential
from layers.dataset import cifar100
(x_train, y_train), (x_test, y_test) = cifar100(1213268041)

test_accuracy=[]
epochs=20
lr=[0.1, 0.01, 0.2]
for i in lr:
    model = Sequential(layers=(FullLayer(32 * 32 * 3, 2500),
                               ReluLayer(),
                               FullLayer(2500, 2000),
                               ReluLayer(),
                               FullLayer(2000, 1500),
                               ReluLayer(),
                               FullLayer(1500, 1000),
                               ReluLayer(),
                               FullLayer(1000, 500),
                               ReluLayer(),
                               FullLayer(500, 4),
                               SoftMaxLayer()),
                       loss=CrossEntropyLayer())
    finalloss = model.fit(x_train, y_train,epochs=20, lr=i)
    y_pred = model.predict(x_test)
    accuracy=(np.mean(y_test == y_pred))
    print('Accuracy: %.2f' % accuracy)
    np.append(test_accuracy, accuracy)
    plt.plot(range(epochs), finalloss, label=('Learning rate=',i))
    plt.xlabel('epochs')
    plt.ylabel('Training loss')
    plt.title('Training loss Vs Epochs for the CIFAR100 data set')
    plt.legend()
plt.show()
plt.plot(lr, test_accuracy)
plt.title('Test Accuracy Vs Learning Rate for Modified NN')
plt.xlabel('Learning rate')
plt.ylabel('Testing Accuracy')
plt.show()

