import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow import keras
import math
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

tf.keras.callbacks.History()

import os

redwine_path = os.path.join(os.path.dirname(__file__), './../dataset/redwine.csv')
whitewine_path = os.path.join(os.path.dirname(__file__), './../dataset/whitewine.csv')
whitewine_path
redwine = pd.read_csv(redwine_path, delimiter=';')
whitewine = pd.read_csv(whitewine_path, delimiter=';')
redwine["isRed"] = 1
whitewine["isRed"] = 0
wine = redwine.append(whitewine, ignore_index=True)
x = wine.iloc[:, :11]
y = np.ravel(wine["isRed"])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=100)

def create_model_for_wine(input_dim, loss_func, hidden_units):
    model = Sequential()
    model.add(Dense(hidden_units, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dense(hidden_units))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer='adam', loss=loss_func, metrics = ["accuracy"]) #note: metrics could also be 'mse'
    
    return model

model = KerasClassifier(create_model_for_wine)  

param_grid = dict(activation_func=['relu'], hidden_units=np.arange(1,100,1), input_dim=[11],
                  loss_func=['binary_crossentropy'],
                  batch_size=[128], epochs=[20])

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10, verbose=10, return_train_score=True)

grid_result = grid.fit(x_train, y_train)

train_scores = grid_result.cv_results_['mean_train_score']
test_scores = grid_result.cv_results_['mean_test_score']

plt.xlabel('hidden units')
plt.ylabel('score')
plt.plot(
    train_scores, label="Training score"
)
plt.plot(
    test_scores, label="Cross-validation score"
)
plt.legend(loc="best")
plt.savefig(os.path.join(os.path.dirname(__file__), "./wine_neural_networks_validation_curve_for_hidden_units.png"), dpi=100)
plt.show()