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

model = SVC()

train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
    model,
    x_train,
    y_train,
    scoring='accuracy',
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    return_times=True,
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
fit_times_std = np.std(fit_times, axis=1)

plt.figure(1)
plt.xlabel('training size')
plt.ylabel('score')
plt.plot(
    train_sizes, train_scores_mean, label="Training score"
)
plt.plot(
    train_sizes, test_scores_mean, label="Cross-validation score"
)
plt.legend(loc="best")
plt.savefig(os.path.join(os.path.dirname(__file__), "./wine_svm_learning_curve_for_training_size.png"), dpi=100)
plt.show()

plt.figure(2)
plt.xlabel('training size')
plt.ylabel('runtime')
plt.plot(
    train_sizes, fit_times_mean, label="Training score"
)
plt.legend(loc="best")
plt.savefig(os.path.join(os.path.dirname(__file__), "./wine_svm_runtime_for_training_size.png"), dpi=100)
plt.show()