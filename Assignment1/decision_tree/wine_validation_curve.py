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

model = DecisionTreeClassifier(max_depth=5)

param_range = np.arange(0.01, 0.08, 0.01)
param_grid = dict(ccp_alpha=param_range)
train_scores, test_scores = validation_curve(model, x_train, y_train, param_name="ccp_alpha", param_range=param_range, n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.xlabel('ccp_alpha')
plt.ylabel('score')
plt.xticks(param_range)
plt.plot(
    param_range, train_scores_mean, label="Training score"
)
plt.plot(
    param_range, test_scores_mean, label="Cross-validation score"
)
plt.legend(loc="best")
plt.savefig(os.path.join(os.path.dirname(__file__), "./wine_decision_tree_validation_curve_for_ccp_alpha.png"), dpi=100)
plt.show()