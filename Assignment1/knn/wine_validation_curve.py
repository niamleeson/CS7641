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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
import os
redwine_path = os.path.join(os.path.dirname(__file__), './../dataset/redwine.csv')
whitewine_path = os.path.join(os.path.dirname(__file__), './../dataset/whitewine.csv')
redwine = pd.read_csv(redwine_path, delimiter=';')
whitewine = pd.read_csv(whitewine_path, delimiter=';')
redwine["isRed"] = 1
whitewine["isRed"] = 0
wine = redwine.append(whitewine, ignore_index=True)
x = wine.iloc[:, :11]
y = np.ravel(wine["isRed"])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=100)

model = KNeighborsClassifier()
param_grid = dict(n_neighbors=[i for i in range(1,100)])
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=3, return_train_score=True)
grid_result = grid.fit(x_train, y_train)

train_scores = grid_result.cv_results_['mean_train_score']
test_scores = grid_result.cv_results_['mean_test_score']

plt.xlabel('number of neighbors')
plt.ylabel('score')
plt.plot(
    train_scores, label="Training score"
)
plt.plot(
    test_scores, label="Cross-validation score"
)
plt.legend(loc="best")
plt.savefig(os.path.join(os.path.dirname(__file__), "./wine_knn_validation_curve_for_number_of_neighbors.png"), dpi=100)
plt.show()