# Importing the libraries to be used:
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm

#csv into dataframe
df = pd.read_csv("mushrooms.csv")
df = df.dropna(axis=1)
print(df.shape)

#converting chars in dataset to ints 
def to_digit(i):
    if i in chars:
        return chars[i]
    else:
        chars[i] = len(chars)+1
        return chars[i]

for idx in list(df.columns):
    chars = {}
    df[idx] = df[idx].map(to_digit)

#Separating X,Y
X = df.loc[ : , df.columns != 'class']
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
print( X_train.shape, X_test.shape)


#removing constant and quasi constant features with sklearn variancethreshold 
sel = VarianceThreshold(threshold=0.1) #threshold = 0 for constant only
sel.fit(X_train)  #finds features w low variance

#get number of features constant (get_support is bool vector rep constant features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
sum(sel.get_support())
print(len([x for x in X_train.columns if x not in X_train.columns[sel.get_support()]]))

# print the features removed 
print([x for x in X_train.columns if x not in X_train.columns[sel.get_support()]])
#result : ['gill-attachment', 'veil-type', 'veil-color', 'ring-number']

X_train = sel.transform(X_train)
X_test = sel.transform(X_test)

print( X_train.shape, X_test.shape)



#linear_svm with L1 reg 
acc_train_svm_linear = []
acc_test_svm_linear = []
c_svm_linear = []

def svm_linear_l1(c):
    svc_linear = svm.LinearSVC(penalty='l1',dual=False,C=c)
    svc_linear.fit(X_train,y_train)
    Yhat_svc_linear_train = svc_linear.predict(X_train)
    acc_train = svc_linear.score(X_train,y_train)
    acc_train_svm_linear.append(acc_train)
    print('Train Accuracy = {0:f}'.format(acc_train))
    Yhat_svc_linear_test = svc_linear.predict(X_test)
    acc_test = svc_linear.score(X_test,y_test)
    acc_test_svm_linear.append(acc_test)
    print('Test Accuracy = {0:f}'.format(acc_test))
    c_svm_linear.append(c)

print("Linear SVM with L1 Reg")
cVals = [0.0001, 0.001, 0.01, 0.1, 1, 10,100]
for c in cVals:
    print(c)
    svm_linear_l1(c)


plt.plot(c_svm_linear,acc_train_svm_linear)
plt.plot(c_svm_linear,acc_test_svm_linear)


# Use the following function to have a legend
plt.legend(['Training Accuracy', 'Test Accuracy'], loc='lower right')
# plt.show()

#linear_svm with L2 reg 
acc_train_svm_linear = []
acc_test_svm_linear = []
c_svm_linear = []

def svm_linear_l2(c):
    svc_linear = svm.SVC(probability = False, kernel = 'linear', C = c)    
    svc_linear.fit(X_train,y_train)
    Yhat_svc_linear_train = svc_linear.predict(X_train)
    acc_train = svc_linear.score(X_train,y_train)
    acc_train_svm_linear.append(acc_train)
    print('Train Accuracy = {0:f}'.format(acc_train))
    Yhat_svc_linear_test = svc_linear.predict(X_test)
    acc_test = svc_linear.score(X_test,y_test)
    acc_test_svm_linear.append(acc_test)
    print('Test Accuracy = {0:f}'.format(acc_test))
    c_svm_linear.append(c)

print("Linear SVM with L2 Reg")
cVals = [0.0001, 0.001, 0.01, 0.1, 1, 10,100]
for c in cVals:
    print(c)
    svm_linear_l2(c)


plt.plot(c_svm_linear,acc_train_svm_linear)
plt.plot(c_svm_linear,acc_test_svm_linear)


# Use the following function to have a legend
plt.legend(['Training Accuracy', 'Test Accuracy'], loc='lower right')
# plt.show()



#radial basis
acc_train_svm_rbf = []
acc_test_svm_rbf = []
c_svm_rbf = []

def svm_rbf(c):
    svc_rbf = svm.SVC(probability = False, kernel = 'rbf', C=c)
    svc_rbf.fit(X_train,y_train)
    Yhat_svc_rbf_train = svc_rbf.predict(X_train)
    acc_train = svc_rbf.score(X_train,y_train)
    acc_train_svm_rbf.append(acc_train)
    print('Train Accuracy = {0:f}'.format(acc_train))
    Yhat_svc_rbf_test = svc_rbf.predict(X_test)
    acc_test = svc_rbf.score(X_test,y_test)
    acc_test_svm_rbf.append(acc_test)
    print('Test Accuracy = {0:f}'.format(acc_test))
    c_svm_rbf.append(c)

print("RBF SVM with L2 Reg")
for c in cVals:
    print(c)
    svm_rbf(c)

plt.plot(c_svm_rbf,acc_train_svm_rbf)
plt.plot(c_svm_rbf,acc_test_svm_rbf)


# Use the following function to have a legend
plt.legend(['Training Accuracy', 'Test Accuracy'], loc='lower right')
# plt.show()


