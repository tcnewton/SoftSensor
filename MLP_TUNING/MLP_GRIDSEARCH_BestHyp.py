# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

# %% [markdown]
# ## MLP CLASSIFIER

# %%
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


# %%
y = np.ravel(y_train)
print(y.shape)


# %%
mlp_rs = MLPClassifier(max_iter=500)
parameter_space = {
    'hidden_layer_sizes': [(256,128,64,32),(1024,512,256,128,64,32),(2056,1024,512)],
    'activation': ['tanh', 'relu','logistic'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
    'batch_size':[10,50],
}
clf = GridSearchCV(mlp_rs, parameter_space, n_jobs=-1, cv=5)
clf.fit(X_trainScaled, y) # X is train samples and y is the corresponding labels


# %%
print('Best parameters found:\n', clf.best_params_)


# %%
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

with open('./pickleSave/clf.p', 'wb') as f:
      pickle.dump(clf, f)

# %%
y_teste = np.ravel(y_test)
y_true, y_pred = y_teste , clf.predict(X_testScaled)
from sklearn.metrics import classification_report
print('Results on the test set:')
print(classification_report(y_true, y_pred))