# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle


# %%
main_path = '../overfitting/setTraining/select_noOhe/'
train = np.load(main_path+'train.npy')
test = np.load(main_path+'test.npy')
y_train = np.load(main_path+'y_train.npy')
y_test = np.load(main_path+'y_test.npy')
print('shapes: train = {}, y_train = {}, test = {}, y_teste = {}'.format(train.shape,y_train.shape,test.shape,y_test.shape))


# %%
# standardscaler x data
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_trainScaled = sc_x.fit_transform(train)
X_testScaled = sc_x.fit_transform(test)


# %%
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
y = np.ravel(y_train)
y_test = np.ravel(y_test)
print(y.shape)


# %%
clf = MLPClassifier(solver="adam",learning_rate="adaptive",hidden_layer_sizes=(12,12,12),
batch_size=10,alpha=0.0001,activation="tanh",max_iter=500,random_state=1).fit(X_trainScaled, y)
y_pred=clf.predict(X_testScaled)
print(clf.score(X_testScaled, y_test))

# %% [markdown]
# # Using best hyperparameters
# Best parameters found:
#  {'activation': 'relu', 'alpha': 0.0001, 'batch_size': 50, 'hidden_layer_sizes': (256, 128, 64, 32), 'learning_rate': 'adaptive', 'power_t': 0.5, 'solver': 'sgd'}

# %%
clf = MLPClassifier(solver="sgd",learning_rate="adaptive",hidden_layer_sizes=(256, 128, 64, 32),
batch_size=50,alpha=0.0001,activation="relu",max_iter=500,random_state=1).fit(X_trainScaled, y)
y_pred=clf.predict(X_testScaled)
print(clf.score(X_testScaled, y_test))

with open('./pickleSave/clf.p', 'wb') as f:
      pickle.dump(clf, f)
