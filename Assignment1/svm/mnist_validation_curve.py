import time
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import os
mnist_path = os.path.join(os.path.dirname(__file__), './../dataset/mnist.csv')
mnist = pd.read_csv(mnist_path)
x = mnist.iloc[:, 1:]
y = mnist.iloc[:, :1]

scaling = MinMaxScaler(feature_range=(0,1)).fit(x)
x = scaling.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=100)

model = SVC()
param_grid = dict(kernel=['linear', 'poly', 'rbf', 'sigmoid'])
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=3, return_train_score=True)
grid_result = grid.fit(x_train, y_train)

train_scores = grid_result.cv_results_['mean_train_score']
test_scores = grid_result.cv_results_['mean_test_score']

print(grid_result.cv_results_['mean_train_score'])