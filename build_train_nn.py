#Import needed packages
import numpy as np
import keras
import tensorflow
from keras import Input
from keras.models import Sequential
from keras.layers import Dense
from keras.models import clone_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Function to build NN to be used for regression
def buildRegressionNN(num_inputs, layer_neurs, activ):

    #define initial model and input layer
    model = Sequential()
    model.add(Input((num_inputs,)))

    #Use for loop to build out hidden layers in the neural network
    for num_neurs in layer_neurs:
        model.add(Dense(num_neurs, activation = activ))

    #Add output layer and compile model
    model.add(Dense(1))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_squared_error'])

    return model

#Function to train and evaluate the model's performance based on split train/test data
def trainAndEvaluate(model, X_train, y_train, X_test, y_test, X_validation, y_validation, epochs):

    #Fit the model to the training and validation data
    model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = epochs, verbose = 0)

    #Score the model and return the model with the score
    predictions = model.predict(X_validation, verbose = 0)
    mse = mean_squared_error(y_validation, predictions)

    return model, mse

#Function to split data between train, test, and validation
def dataSplit(features, target, train_per, test_per, val_per):

    #perform the split between training data and all other data
    X_train, X_evaluation, y_train, y_evaluation = train_test_split(features, target, test_size = 1 - train_per)
    
    #perform the split between test and validation data
    X_test, X_val, y_test, y_val = train_test_split(X_evaluation, y_evaluation, test_size = val_per/(test_per + val_per))

    #return split dataset
    return X_train, y_train, X_test, y_test, X_val, y_val