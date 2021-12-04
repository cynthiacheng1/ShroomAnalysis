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

corr_data = df.copy()

#It's not important what character is what number for calculate. Just turn it into digit value.
def to_digit(i):
    if i in chars:
        return chars[i]
    else:
        chars[i] = len(chars)+1
        return chars[i]
    
features = list(corr_data.columns)

for idx in features:
    chars = {}
    corr_data[idx] = corr_data[idx].map(to_digit)

print(corr_data)

df = corr_data

X = df.loc[ : , df.columns != 'class']
y = np.array(df['class'])

print("shape of X: ", X.shape)
print("shape of y: ", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

print(X_train.shape, X_test.shape)

# using sklearn variancethreshold to find constant features
sel = VarianceThreshold(threshold=0)
sel.fit(X_train)  # fit finds the features with zero variance

# get_support is a boolean vector that indicates which features are retained
# if we sum over get_support, we get the number of features that are not constant
sum(sel.get_support())

# alternate way of finding non-constant features
len(X_train.columns[sel.get_support()])

# print the constant features
print(
    len([
        x for x in X_train.columns
        if x not in X_train.columns[sel.get_support()]
    ]))

[x for x in X_train.columns if x not in X_train.columns[sel.get_support()]]

X_train = sel.transform(X_train)
X_test = sel.transform(X_test)

print( X_train.shape, X_test.shape)