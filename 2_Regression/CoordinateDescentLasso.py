'''
Created on 20-Nov-2016
Programming Assignment 1 of Week4 of 
Regression Course of Machine Learning from Coursera
@author: sudheer
'''
import numpy as np
import pandas as pd
import math

def get_numpy_data(data_frame, features,output):
    features_matrix = data_frame[features].as_matrix()
    ones = np.ones( (np.shape(features_matrix)[0], 1) )
    features_matrix = np.column_stack((ones,features_matrix))
    output_matrix = data_frame[[output]].as_matrix()[:,0]
    return(features_matrix,output_matrix)

def predict_output(feature_matrix,weights):
    return np.dot(feature_matrix,weights)


def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix,axis=0)
    feature_matrix_normalized = feature_matrix/norms
    return(feature_matrix_normalized,norms)

def lasso_coordinate_descent_step(i,feature_matrix,output,weights,l1_penalty):
    prediction = predict_output(feature_matrix, weights)
    feature_i = feature_matrix[:,i]
    ro_i = np.sum(feature_i * (output - prediction + weights[i]*feature_i))
    if i == 0:
        new_weight_i = ro_i
    elif ro_i < -l1_penalty/2.:
        new_weight_i = ro_i + l1_penalty/2.
    elif ro_i > l1_penalty/2.:
        new_weight_i = ro_i - l1_penalty/2.
    else:
        new_weight_i = 0
    return new_weight_i

def lasso_cyclic_coordinate_descent(feature_matrix,output,initial_weights,l1_penalty,tolerance):
    converged = False
    weights = np.copy(initial_weights)  #weights = initial_weights will do a copy by reference and might lead to wrong values V.V.IMP
    while not converged:
        #Maintain an array of how much each weight changed
        weightDiffs = np.empty(len(weights))
        #weights_after_single_step = np.empty(len(weights))
        for i in range(len(weights)):
            new_weight_i = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
            weightDiffs[i] = abs(new_weight_i-weights[i])
            weights[i] = new_weight_i
            #weights_after_single_step[i] = new_weight_i
        #weights = weights_after_single_step
        if (all(i < tolerance for i in weightDiffs)):
            converged = True
            return weights
        

sales = pd.read_csv('../Data/Week5/kc_house_data.csv')
features = ['sqft_living','bedrooms']
output = 'price'

(feature_matrix,output) = get_numpy_data(sales, features, output)
(normalized_features,norms) = normalize_features(feature_matrix)
weights = np.array([1,4,1])
predictions = predict_output(normalized_features, weights)

#Quiz Q1:
ro = np.empty((np.shape(weights)[0],1))
for i in range(len(weights)):
    feature_i = normalized_features[:,i]   
    ro[i] = np.sum(feature_i * (output - predictions + weights[i]*feature_i ))
print('PRINTING RO: ')
print(ro)

#Quiz Q2:
weights = [0.,0.,0.]
l1_penalty = 1e7
tolerance = 1.0
weights = lasso_cyclic_coordinate_descent(normalized_features, output, weights, l1_penalty, tolerance)
predictions = predict_output(normalized_features, weights)
RSS = np.sum(np.power(np.subtract(output,predictions),2))
print('RSS with sqft_living and bedrooms: ' + str( RSS))

#Quiz Q3:
print('Printing weights to identify features having non-zero convergence: ')
print(weights)

trainData = pd.read_csv('../Data/Week5/kc_house_train_data.csv')
testData = pd.read_csv('../Data/Week5/kc_house_test_data.csv')

features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition',
            'grade','sqft_above','sqft_basement','yr_built','yr_renovated']

(feature_matrix,output) = get_numpy_data(trainData, features, 'price')
(normalized_features,norms) = normalize_features(feature_matrix)

#Quiz Q4:
l1_penalty = 1e7
initial_weights = np.zeros(len(features)+1)
weights1e7 = np.empty(len(features)+1)
tolerance = 1.0
weights1e7 = lasso_cyclic_coordinate_descent(normalized_features, output, initial_weights, l1_penalty, tolerance)
print('weights1e7: ')
print(weights1e7)

#Quiz Q5:
l1_penalty = 1e8
weights1e8 = np.empty(len(features)+1)
tolerance = 1.0
weights1e8 = lasso_cyclic_coordinate_descent(normalized_features, output, initial_weights, l1_penalty, tolerance)
print('weights1e8: ')
print(weights1e8)

#Quiz Q6:
l1_penalty = 1e4
weights1e4 = np.empty(len(features)+1)
tolerance = 5e5
weights1e4 = lasso_cyclic_coordinate_descent(normalized_features, output, initial_weights, l1_penalty, tolerance)
print('weights1e4: ')
print(weights1e4)

weights1e7 = np.divide(weights1e7,norms)
weights1e8 = np.divide(weights1e8,norms)
weights1e4 = np.divide(weights1e4,norms)

print(weights1e7[3])

(feature_matrix,output) = get_numpy_data(testData,features,'price')

predictions_1e7 = predict_output(feature_matrix, weights1e7)
RSS_1e7 = np.sum(np.power(np.subtract(predictions_1e7,output),2))
print('RSS_1e7: ' + str(RSS_1e7))

predictions_1e8 = predict_output(feature_matrix, weights1e8)
RSS_1e8 = np.sum(np.power(np.subtract(predictions_1e8,output),2))
print('RSS_1e8: ' + str(RSS_1e8))

predictions_1e4 = predict_output(feature_matrix, weights1e4)
RSS_1e4 = np.sum(np.power(np.subtract(predictions_1e4,output),2))
print('RSS_1e4: ' + str(RSS_1e4))




