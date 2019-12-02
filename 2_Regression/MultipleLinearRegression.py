'''
Created on 21-Aug-2016
Programming Assignment 2 for Week 2 
of Regression course of Machine Learning
Specialization of Coursera.
@author: sudheer
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_numpy_data(data_frame, features, output):
    features_matrix = data_frame[features].as_matrix() #convert the pandas dataframe to numpy matrix
    ones = np.ones( (np.shape(features_matrix)[0] , 1) ) #create a column of ones
    features_matrix = np.column_stack( (ones,features_matrix) ) #append the column of ones to features_matrix
    output_matrix = data_frame[[output]].as_matrix()
    output_matrix = output_matrix[:,0]
    #print 'size of output_matrix from get_numpy_data is ', np.shape(output_matrix)
    return (features_matrix,output_matrix)

def predict_outcome(features_matrix,weights):
    return np.dot(features_matrix,weights)

def feature_derivative(error, feature):
    #print(np.shape(error))
    #print(np.dot(error,feature))
    return 2 * np.dot(error,feature)

def regression_gradient_descent(feature_matrix,output,initial_weights,step_size,tolerance):
    converged = False
    weights = np.array(initial_weights)
    
    while not converged:
        gradient_sum_squares = 0
        predicted = predict_outcome(feature_matrix, weights)
        #print 'size of predicted is ', np.shape(predicted)
        #output = output[:,0]
        #print 'size of output is ' , np.shape(output)
        error = output - predicted
        for i in range(len(weights)):
            derivativeI = feature_derivative(error, feature_matrix[:,i])
            gradient_sum_squares += derivativeI ** 2
            #print(derivativeI)
            weights[i] += step_size * derivativeI
        
        gradient_magnitude = np.sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
        
    return weights

train_data = pd.read_csv('../Data/Week2/kc_house_train_data.csv')
test_data = pd.read_csv('../Data/Week2/kc_house_test_data.csv')

simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

simple_weights = regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, tolerance)

#Quiz question1
print 'A1: weight coefficient for sqft is ', simple_weights[1] 

#Quiz question2
(test_simple_feature_matrix,test_output) = get_numpy_data(test_data, simple_features, my_output)
predictions_test_data = predict_outcome(test_simple_feature_matrix, simple_weights)
print 'A2: The predicted value of house 1 in test set according to the model1 is ' , predictions_test_data[0]

#Quiz question3
#Compute RSS for the test data
error_model1 = predictions_test_data - test_output
error_squared = np.apply_along_axis(lambda x: x**2, 0, error_model1)
RSS = np.sum(error_squared)
print 'A3: RSS on the test set using model1 is ', RSS

#Quiz question4
#Construct a second model using 'sqft_living' and 'sqft_living_15'
model2_features = ['sqft_living','sqft_living15']
(feature_matrix,output) = get_numpy_data(train_data,model2_features,my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9
#Compute the weights for model2
model2_weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)

#compute the predictions for the test data using model2 weights
(test_model2_feature_matrix,test_output) = get_numpy_data(test_data, model2_features, my_output)
predictions_test_data = predict_outcome(test_model2_feature_matrix, model2_weights)
print 'A4: The predicted value of house 1 in test set according to the model2 is ', predictions_test_data[0]

print 'the actual value of house 1 in test set is ', test_output[0]

#Quiz question5
#Compute RSS on the test data
error_model2 = predictions_test_data - test_output
error_squared = np.apply_along_axis(lambda x: x**2,0,error_model2)
RSS = np.sum(error_squared)
print 'A5: RSS on test set using model2 is ', RSS


#A = np.array([1,2,3])
#RSS = np.apply_along_axis(lambda x: x**2, 0, A)
#print np.sum(RSS,0)   
#A = np.array([[1,2],[2,1]])
#ones = np.ones((np.shape(A)[0],1))
#print(ones)
#A = np.column_stack(( ones , A ) )
#print(A)
#(feature_matrix,output) = get_numpy_data(train_data, model_features, my_output)
#print(feature_matrix[1])
