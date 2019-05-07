# Logistic regression predicts discrete values, meaning that it can only
# output either 1 or 0. For this reason, logistic regression is used for
# classification problems, i.e. predicting wether someone has diabetes or not

import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

costs= [];

def hypothesis(row, coefficients):
    acum = 0
    for i in range(len(coefficients)):
        acum += coefficients[i] * row[i]  #evaluates h(x) = a+bx1+cx2+ ... nxn..
    return 1.0 / (1.0 + math.exp(-acum))  # Sigmoid function

def gradient_descent(data, learning_rate, coefficients):
    temp_coefficients = np.zeros(len(coefficients))
    for i in range(len(coefficients)):
        acum = 0
        for j in range(len(data)):
            prediction = hypothesis(data[j], coefficients)
            acum += (prediction - data[j][-1]) * data[j][i]
        temp_coefficients[i] = coefficients[i] - (learning_rate * (1 / len(data))) * acum
    return temp_coefficients

def cross_entropy(data, coefficients):
    acum = 0
    for i in range(len(data)):
        prediction = hypothesis(data[i], coefficients)
        if data[i][-1] == 1:   # When y == 1
            if prediction == 0:
                prediction = 0.000001  # Value close to 0
            error = math.log(prediction) * -1
        else:   # When y == 0
            if prediction == 1:
                prediction = 0.999999   # Attempting to create a value close to 0
            error = math.log(1-prediction) * -1
        acum += error
    c = acum / len(data)
    costs.append(c)
    return c

# Hyperparameters
learning_rate =.03  #  learning rate
epochs = 1000 # epochs
filename = "train.csv"


# For this case, all of the data from the training set was used, so the
# the size of the batch was of 891 rows, with 6 columns Bias, Pclass, Sex, Age, Fare, and Embarked
data = np.genfromtxt(filename, delimiter=',')   # Reading training set
coefficients = np.zeros(len(data[0]))   # Creation of coefficients or thetas

data = np.insert(data, 0, 1, axis=1)   # Inserting a new column for the bias

for epoch in range(epochs):
    temp = coefficients
    coefficients = gradient_descent(data, learning_rate, coefficients)
    error = cross_entropy(data, coefficients)
    if error < 0.001:
        break

print("Coefficients: ", coefficients)

'''
testfile = "test.csv"
test_with_id = np.genfromtxt(testfile, delimiter=',')
test_set = np.delete(test_with_id,[0], axis=1) # Dropping the ID column
test_set = np.insert(test_set, 0, 1, axis=1)

with open('test_submission.csv', 'w') as test_submission:
    test_submission.write("PassengerId,Survived\n")
    for i in range(len(test_set)):
        prediction = hypothesis(test_set[i], coefficients)
        if prediction > 0.5:
            test_submission.write(str(int(test_with_id[i][0])) + ",1\n")
        else:
            test_submission.write(str(int(test_with_id[i][0])) + ",0\n")
'''
