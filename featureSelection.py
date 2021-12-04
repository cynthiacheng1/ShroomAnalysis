import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split  # The standard - train/test to prevent overfitting and choose hyperparameters
from sklearn.feature_selection import VarianceThreshold

import warnings
warnings.filterwarnings('ignore')

# import the Santander customer satisfaction dataset from Kaggle

df = pd.read_csv("mushrooms.csv")

#It's not important what character is what number for calculate. Just turn it into digit value.
def to_digit(i):
    if i in chars:
        return chars[i]
    else:
        chars[i] = len(chars)+1
        return chars[i]
    
features = list(df.columns)

for idx in features:
    chars = {}
    df[idx] = df[idx].map(to_digit)


X = df.loc[ : , df.columns != 'class']
y = np.array(df['class'])

print("shape of X: ", X.shape)
print("shape of y: ", y.shape)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

X_train = X[:4874]
X_test = X[4874:]

print(X_train.shape, X_test.shape)

#removing constant and quasi constant features
# using sklearn variancethreshold to find constant features

sel = VarianceThreshold(threshold=0.01) #threshold = 0 for constant only
sel.fit(X_train)  # fit finds the features with zero variance

# get_support is a boolean vector that indicates which features are retained
# if we sum over get_support, we get the number of features that are not constant
X_train = X[:4874]

sum(sel.get_support())
print(len([x for x in X_train.columns if x not in X_train.columns[sel.get_support()]]))

# print the constant features
print([x for x in X_train.columns if x not in X_train.columns[sel.get_support()]])

X_train = sel.transform(X_train)
X_test = sel.transform(X_test)

print( X_train.shape, X_test.shape)

print(X_train)