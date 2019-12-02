'''
Created on 16-Oct-2016
Programming Assignment 2 for Week 4
of Regression Course of Machine Learning
Specialization of Coursera
@author: sudheer
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_numpy_data(data_frame, features, output):
    features_matrix = data_frame[features].as_matrix()
    ones = np.ones( (np.shape(features_matrix)[0], 1) )
    features_matrix = np.column_stack((ones,features_matrix))
    output_matrix = data_frame[[output]].as_matrix()
    output_matrix = output_matrix[:,0]
    return (features_matrix,output_matrix)

def predict_output(feature_matrix, weights):
    predictions = np.dot(feature_matrix,weights)
    return predictions

def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    if not feature_is_constant:
        return 2 * np.dot(errors, feature)
    return 2 * np.dot(errors, feature) + 2 * l2_penalty * weight

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
sales = pd.read_csv('../Data/Week4/kc_house_data.csv',dtype = dtype_dict)

(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price')

def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100):
    weights = np.array(initial_weights)#[:,0]
    current_iter = 0
    while current_iter <= max_iterations:
        predictions = predict_output(feature_matrix, weights)
        errors = predictions-output
        for i in range(len(weights)):
            derivativeI = feature_derivative_ridge(errors, feature_matrix[:,i], weights[i], l2_penalty, i)
            weights[i] = weights[i] - step_size * derivativeI
        current_iter = current_iter+1
    return weights

train_data = pd.read_csv('../Data/Week4/kc_house_train_data.csv', dtype=dtype_dict)
test_data = pd.read_csv('../Data/Week4/kc_house_test_data.csv', dtype=dtype_dict)

simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix,output) = get_numpy_data(train_data, simple_features, my_output)
(simple_test_feature_matrix,test_output) = get_numpy_data(test_data, simple_features, my_output)

step_size = 1e-12
max_iterations = 1000
initial_weights = np.zeros( (np.shape(simple_feature_matrix)[1], 1) )[:,0]

#Observe the weights without regularization
l2_penalty = 0
simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights , step_size, l2_penalty, max_iterations)
print("Simple_Weights_0_Penalty " + str(simple_weights_0_penalty))

#Observe the weights with a high l2 penalty
l2_penalty = 1e11
simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations)
print("Simple_Weights_High_Penalty " + str(simple_weights_high_penalty))

'''
Compute the RSS on the test data for the following weights:
1. Initial weights (All zeros)
2. Weights learnt with no regularization (simple_weights_0_penalty)
3. Weights learnt with high regularization (simple_weights_high_penalty)
'''
RSS_test_data_simple_initial_weights = np.sum(np.apply_along_axis(lambda x: x**2, 0, predict_output(simple_test_feature_matrix,initial_weights)-test_output))
RSS_test_data_simple_0_penalty = np.sum(np.apply_along_axis(lambda x: x**2, 0, predict_output(simple_test_feature_matrix,simple_weights_0_penalty)-test_output))
RSS_test_data_simple_high_penalty = np.sum(np.apply_along_axis(lambda x: x**2, 0, predict_output(simple_test_feature_matrix,simple_weights_high_penalty)-test_output))
print("RSS_test_data_simple_initial_weights "+ str(RSS_test_data_simple_initial_weights)+"\n"+
      "RSS_test_data_simple_0_penalty "+ str(RSS_test_data_simple_0_penalty)+ "\n"+
      "RSS_test_data_simple_high_penalty "+ str(RSS_test_data_simple_high_penalty))

'''Now consider a model with two features
    sqft_living and sqft_living15
'''
model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)

step_size = 1e-12
max_iterations = 1000
initial_weights = np.zeros( (np.shape(feature_matrix)[1], 1) )[:,0]  

l2_penalty = 0
multiple_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations)
print("Multiple_Weights_0_Penalty " + str(multiple_weights_0_penalty))

l2_penalty = 1e11
multiple_weights_high_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations)
print("Multiple_Weights_High_Penalty " + str(multiple_weights_high_penalty))

'''
Compute the RSS on the test data for the following weights:
1. Initial weights (All zeros)
2. Weights learnt with no regularization (multiple_weights_0_penalty)
3. Weights learnt with high regularization (multiple_weights_high_penalty)
'''
RSS_test_data_multiple_initial_weights = np.sum(np.apply_along_axis(lambda x: x**2, 0, predict_output(test_feature_matrix,initial_weights)-test_output))
RSS_test_data_multiple_0_penalty = np.sum(np.apply_along_axis(lambda x: x**2, 0, predict_output(test_feature_matrix,multiple_weights_0_penalty)-test_output))
RSS_test_data_multiple_high_penalty = np.sum(np.apply_along_axis(lambda x: x**2, 0, predict_output(test_feature_matrix,multiple_weights_high_penalty)-test_output))

print("RSS_test_data_multiple_initial_weights "+ str(RSS_test_data_multiple_initial_weights)+"\n"+
      "RSS_test_data_multiple_0_penalty "+ str(RSS_test_data_multiple_0_penalty)+ "\n"+
      "RSS_test_data_multiple_high_penalty "+ str(RSS_test_data_multiple_high_penalty))

'''Predict the house price for the 1st house in the test set using 
the no regularization and high regularization models.
'''
first_house = test_feature_matrix[0]
error_first_house_predicted_val_no_reg = test_output[0]-predict_output(first_house,multiple_weights_0_penalty)
error_first_house_predicted_val_high_reg = test_output[0]-predict_output(first_house, multiple_weights_high_penalty)
print("\n" + "first_house_predicted_val_no_reg " + str(error_first_house_predicted_val_no_reg) +
      "\n" + "first_house_predicted_val_high_reg " + str(error_first_house_predicted_val_high_reg))

'''
my_weights = np.array([1.,10.])
test_predictions = predict_output(example_features, my_weights)
errors = test_predictions - example_output
print (feature_derivative_ridge(errors, example_features[:,1], my_weights[1], 1, False))
print (np.sum(errors*example_features[:,1])*2 + 20)'''

