import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')

#csv into dataframe
df = pd.read_csv("mushrooms.csv")

#converting chars in dataset to ints 
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

#Separating X,Y
X = df.loc[ : , df.columns != 'class']
y = np.array(df['class'])
# print("shape of X: ", X.shape)
# print("shape of y: ", y.shape)

#splitting dataset (about 40 percent split)
X_train = X[:4874]
X_test = X[4874:]
print(X_train.shape, X_test.shape)

#removing constant and quasi constant features with sklearn variancethreshold 
sel = VarianceThreshold(threshold=0.01) #threshold = 0 for constant only
sel.fit(X_train)  #finds features w low variance

#get number of features constant (get_support is bool vector rep constant features)
X_train = X[:4874]
sum(sel.get_support())
print(len([x for x in X_train.columns if x not in X_train.columns[sel.get_support()]]))

# print the features removed 
print([x for x in X_train.columns if x not in X_train.columns[sel.get_support()]])
#result : ['gill-attachment', 'veil-type', 'veil-color', 'ring-number']

X_train = sel.transform(X_train)
X_test = sel.transform(X_test)

print( X_train.shape, X_test.shape)

# print(X_train)