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
mnist_path = os.path.join(os.path.dirname(__file__), './../dataset/mnist.csv')
mnist = pd.read_csv(mnist_path)
x = mnist.iloc[:, 1:]
x = x.astype('float32') / 255

y = mnist.iloc[:, :1]
num_labels = len(np.unique(y))
y = to_categorical(y)

image_size = x.shape[1]

print(image_size, num_labels)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=100)

def create_model_for_mnist(activation_func, loss_func, drop_out, hidden_units):
    model = Sequential()
    model.add(Dense(hidden_units, input_dim=image_size))
    model.add(Activation(activation_func))
    model.add(Dropout(drop_out))
    model.add(Dense(hidden_units))
    model.add(Activation(activation_func))
    model.add(Dropout(drop_out))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss=loss_func, metrics = ["accuracy"]) #note: metrics could also be 'mse'
    
    return model

model = KerasClassifier(build_fn=create_model_for_mnist, hidden_units=128, drop_out=0.4,
                  activation_func='relu', loss_func='binary_crossentropy',
                  batch_size=128, epochs=5, validation_split=0.3, verbose = False)

train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
    model,
    x_train,
    y_train,
    # scoring='accuracy',
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
plt.savefig(os.path.join(os.path.dirname(__file__), "./mnist_neural_networks_learning_curve_for_training_size.png"), dpi=100)
plt.show()

plt.figure(2)
plt.xlabel('training size')
plt.ylabel('runtime')
plt.plot(
    train_sizes, fit_times_mean, label="Training score"
)
plt.legend(loc="best")
plt.savefig(os.path.join(os.path.dirname(__file__), "./mnist_neural_networks_runtime_for_training_size.png"), dpi=100)
plt.show()