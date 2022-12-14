# CS7641

## Assignment 1 Instructions

Please clone the repo:

```
git clone https://github.com/niamleeson/CS7641.git

cd ./CS7641
```

The code for each experiement was split in consideration of long execution time. Please let me know if you'd like the experiments to be combined in the future.

Please run the following commands to install the dependencies:

```
cd ./Assignment1

pip install -r requirements.txt
```

To run each experiment, please choose one of the following commands:

### Decision Tree 
- validation curve based on different pruning configuration value
  - MNIST dataset: `python .\decision_tree\mnist_validation_curve.py`
  - wine dataset: `python .\decision_tree\wine_validation_curve.py`
- learning curve and runtime based on different training dataset size
  - MNIST dataset: `python .\decision_tree\mnist_learning_curve.py`
  - wine dataset: `python .\decision_tree\wine_learning_curve.py`

### Boosting
- validation curve based on different number of weak learners
  - MNIST dataset: `python .\boosting\mnist_validation_curve.py`
  - wine dataset: `python .\boosting\wine_validation_curve.py`
- learning curve and runtime based on different training dataset size
  - MNIST dataset: `python .\boosting\mnist_learning_curve.py`
  - wine dataset: `python .\boosting\wine_learning_curve.py`

### Neural Networks
- validation curve based on different number of hidden units
  - MNIST dataset: `python .\neural_networks\mnist_validation_curve.py`
  - wine dataset: `python .\neural_networks\wine_validation_curve.py`
- learning curve and runtime based on different training dataset size
  - MNIST dataset: `python .\neural_networks\mnist_learning_curve.py`
  - wine dataset: `python .\neural_networks\wine_learning_curve.py`

### SVM
- accuracy of each of the four kernel types
  - MNIST dataset: `python .\svm\mnist_validation_curve.py`
  - wine dataset: `python .\svm\wine_validation_curve.py`
- learning curve and runtime based on different training dataset size
  - MNIST dataset: `python .\svm\mnist_learning_curve.py`
  - wine dataset: `python .\svm\wine_learning_curve.py`

### KNN
- validation curve based on different number of nearest neighbors
  - MNIST dataset: `python .\knn\mnist_validation_curve.py`
  - wine dataset: `python .\knn\wine_validation_curve.py`
- learning curve and runtime based on different training dataset size
  - MNIST dataset: `python .\knn\mnist_learning_curve.py`
  - wine dataset: `python .\knn\wine_learning_curve.py`
