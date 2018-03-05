from svm import SVM
import numpy as np
import matplotlib.pyplot as plt
import datasets
from matplotlib import pylab
import sklearn.svm
import csv


# load data
x_train, y_train, x_test, y_test = datasets.moon_dataset(n_train=500, n_test=500)
#########################
model = sklearn.svm.SVC(C=4.0, kernel='linear')
model.fit(x_train, y_train)
y_pred_lin = model.predict(x_test)

from sklearn.metrics import accuracy_score
a_lin = accuracy_score(y_test, y_pred_lin)
print('Accuracy_linear: %.2f' % a_lin)
###########################
model = sklearn.svm.SVC(C=2.0, kernel='poly', degree=5,coef0=3)
model.fit(x_train, y_train)
y_pred_poly = model.predict(x_test)

from sklearn.metrics import accuracy_score
a_poly = accuracy_score(y_test, y_pred_poly)
print('Accuracy_poly: %.2f' % a_poly)
###########################
model = sklearn.svm.SVC(C=2.0, kernel='rbf')
model.fit(x_train, y_train)
y_pred_rbf = model.predict(x_test)

from sklearn.metrics import accuracy_score
a_rbf=accuracy_score(y_test, y_pred_rbf)
print('Accuracy_rbf: %.2f' % a_rbf)
###########################
model = sklearn.svm.SVC(C=8.0, kernel='sigmoid', coef0=4)
model.fit(x_train, y_train)
y_pred_sig = model.predict(x_test)

from sklearn.metrics import accuracy_score
a_sig = accuracy_score(y_test, y_pred_sig)
print('Accuracy_sigmoid: %.2f' % a_sig)

toCSV = [{'Algorithm':'linear','Accuracy':a_lin}, {'Algorithm':'polynomial','Accuracy':a_poly}, {'Algorithm':'rbf','Accuracy':a_rbf},{'Algorithm':'sigmoid','Accuracy':a_sig}]
keys = toCSV[0].keys()
with open('svm results.csv', 'wb') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(toCSV)