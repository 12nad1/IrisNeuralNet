# Maxwell Kaye
# Comp 131 Problem Set 4
# Dec 7, 2018
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split


# The sigmoid function for calculating the output of each neuron
def sigmoid(p):
    return 1/(1+np.exp(-p))


# The derivative of the sigmoid function
def sigprime(p):
    return sigmoid(p) * (1 - sigmoid(p))


# Normalize array
def normalize(X):
    l2 = np.atleast_1d(np.linalg.norm(X, 2, -1))
    l2[l2 == 0] = 1
    normX = X / np.expand_dims(l2, -1)
    return normX


class NeuralNet:
    def __init__(self):
        self.weight1 = 0.499 * np.random.random((4, 4)) + .001  # weights from input layer to hidden layer
        self.weight2 = 0.499 * np.random.random((4, 3)) + .001  # weights from hidden layer to output layer
        self.learning_rate = 0.1  # learning rate

    def predict(self, features):
        # Hidden Layer
        potential1 = np.dot(features, self.weight1)
        output1 = sigmoid(potential1)

        # Output Layer
        potential2 = np.dot(output1, self.weight2)
        output2 = sigmoid(potential2)
        return output2

    # forward propagation, backward propagation, updates weights, loops while validation accuracy increases
    def train(self, train_features, train_labels, val_features, val_labels):

        # variables to check if the previous the current validation accuracy is worse than the previous 1000 accuracies
        improving = True
        accuracies = list(np.zeros(1000))
        ctr = 0

        print('Training', end = ' ')
        sys.stdout.flush()

        while improving:
            # Forward propagation
            input = train_features

            potential1 = np.dot(input, self.weight1)
            output1 = sigmoid(potential1)

            potential2 = np.dot(output1, self.weight2)
            output2 = sigmoid(potential2)

            # Backward propagation using gradient descent
            error2 = train_labels - output2
            delta2 = error2 * sigprime(potential2)

            error1 = np.dot(delta2, self.weight2.T)
            delta1 = error1 * sigprime(potential1)

            self.weight2 += np.dot(output1.T, delta2) * nn.learning_rate
            self.weight1 += np.dot(input.T, delta1) * nn.learning_rate

            val_err = np.mean(np.square(val_labels - self.predict(val_features)))
            val_acc = (1 - val_err) * 100

            if all(val_acc < prev_acc for prev_acc in accuracies[ctr:]) or ctr > 10000:
                improving = False
                train_err = np.mean(np.square(train_labels - self.predict(train_features)))
                print('\nValidation accuracy: ', (1 - val_err) * 100,'%', '  Training accuracy: ', (1 - train_err) * 100,'%')

            accuracies.append(val_acc)
            ctr += 1
            if ctr % 500 == 0:
                print(' .', end = ' ')
                sys.stdout.flush()


# read data from csv
iris = pd.read_csv('IrisData.txt', header = None, names = ['f1', 'f2', 'f3', 'f4', 'labels'])


# Replace the species with 0, 1, 2  (the index of their one hot value)
iris['labels'].replace(['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'], [0, 1, 2], inplace=True)

# Get features
features = pd.DataFrame(iris, columns=['f1', 'f2', 'f3', 'f4'])
features = normalize(features.values)

# Get labels and flatten
labels = pd.DataFrame(iris, columns=['labels'])
labels = labels.values
labels = labels.flatten()

# Convert to one-hot
onehot = np.zeros((len(labels), 3), dtype=int)
for i, lab in enumerate(labels):
    onehot[i][lab] = 1.
labels = onehot

# Split data to training, validation, and testing data (80:15:5)
train_features, val_test_features, train_labels, val_test_labels = train_test_split(features, labels, test_size=0.2)
val_features, test_features, val_labels, test_labels = train_test_split(val_test_features, val_test_labels, test_size=0.25)

nn = NeuralNet()
nn.train(train_features, train_labels, val_features, val_labels)
print('Training Complete.\n')

# convert test labels back to their actual names
new_test_labels = []
for i, label in enumerate(test_labels):
    if label[0] == 1:
        new_test_labels.append('Iris-setosa')
    elif label[1] == 1:
        new_test_labels.append('Iris-virginica')
    else:
        new_test_labels.append('Iris-versicolor')


# user interface
print('Normalized test features (unused in training or validation):')
print(['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Label'])
for i, label in enumerate(new_test_labels):
    print(test_features[i], label)


quit = 'no'
while quit != 'quit':
    print('\nHello Gardner. Please input the features of the flower you would like to predict.')
    print('You may enter normalized data, or data in Cm, as long as it is all consistent.')
    sepal_length = float(input('Sepal Length?'))
    sepal_width = float(input('Sepal Width?'))
    petal_length = float(input('Petal Length?'))
    petal_width = float(input('Petal Width?'))

    # use user input to predict flower
    input_features = np.asarray([sepal_length, sepal_width, petal_length, petal_width])
    input_features = normalize(input_features)
    output = nn.predict(np.asarray(input_features))
    pred = np.argmax(output.flatten())


    # convert prediction to string
    if pred == 0:
        pred = 'Iris-setosa'
    elif pred == 1:
        pred = 'Iris-virginica'
    else:
        pred = 'Iris-versicolor'

    # final results
    print('\nPrediction: ', pred)

    quit = input('\nDo you wish to quit? Type "quit" if so. Type anything else if not.')