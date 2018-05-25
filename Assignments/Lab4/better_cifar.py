import numpy as np
import matplotlib.pyplot as plt

from layers import (FullLayer, ReluLayer, SoftMaxLayer, CrossEntropyLayer, ConvLayer, MaxPoolLayer, FlattenLayer, Sequential)
from layers.full import FullLayer
from layers.relu import ReluLayer
from layers.softmax import SoftMaxLayer
from cross_entropy import CrossEntropyLayer
from conv_new import ConvLayer
from maxpool_new import MaxPoolLayer
from flatten_new import FlattenLayer
from sequential import Sequential
from layers.dataset_new import cifar100
from dataset_new import onehot
(x_train, y_train), (x_test, y_test) = cifar100(1213268041)

test_accuracy=[]
epochs=15
finalloss=[]
testloss=[]
model = Sequential(layers=(ConvLayer(3,16,3),
                               ReluLayer(),
                               MaxPoolLayer(2),
                               ConvLayer(16,32,3),
                               ReluLayer(),MaxPoolLayer(2),
                               FlattenLayer(),
                               FullLayer(8*8*32,4),
                               SoftMaxLayer()),
                       loss=CrossEntropyLayer())

train_loss,valid_loss = model.fit(x_train, y_train,x_test,y_test, epochs=15, lr=0.1, batch_size=128)
y_pred = model.predict(x_test)
accuracy=(np.mean(y_test == onehot(y_pred)))
print('Accuracy: %.2f' % accuracy)
#np.append(test_accuracy, accuracy)
plt.plot(range(len(train_loss)), train_loss, label='Training loss')
plt.plot(range(len(valid_loss)), valid_loss, label='Testing loss')
plt.xlabel('epochs')
plt.ylabel('Training & Testing loss')
plt.title('Training & Testing loss Vs Epochs for the CIFAR100 data set')
plt.legend()
plt.show()
# plt.plot(lr, test_accuracy)
# plt.title('Test Accuracy Vs Learning Rate for Modified NN')
# plt.xlabel('Learning rate')
# plt.ylabel('Testing Accuracy')
# plt.show()

