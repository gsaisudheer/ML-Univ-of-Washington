'''
Created on 27-Nov-2016
Programming Assignment 1 of Week6 of 
Regression Course of Machine Learning from Coursera
@author: sudheer
'''

import numpy as np
import pandas as pd

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

def get_numpy_data(data_frame, features,output):
    features_matrix = data_frame[features].as_matrix()
    ones = np.ones( (np.shape(features_matrix)[0], 1) )
    features_matrix = np.column_stack((ones,features_matrix))
    output_matrix = data_frame[[output]].as_matrix()[:,0]
    return(features_matrix,output_matrix)

def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix,axis=0)
    feature_matrix_normalized = feature_matrix/norms
    return(feature_matrix_normalized,norms)

#Function to compute distances from a query instance to all the given instances
def compute_distances(features_instances, features_query ):
    distances = np.sqrt(np.sum((features_instances - features_query)**2,axis=1))
    return distances

#Perform k-Nearest Neighbors regression
def k_nearest_neighbors(k,features_train,features_query):
    distances = compute_distances(features_train,features_query)
    neighbors = np.argsort(distances)
    return neighbors[0:k]

#Predict the price of a house based on the prices of nearest neighbors
def predict_output_of_query(k,features_train,output_train,features_query):
    neighbors = k_nearest_neighbors(k, features_train, features_query)
    return np.average(output_train[neighbors])

#Write a function that accepts a list of house features and returns the predictions for them
def predict_output(k,features_train,output_train,features_query):
    num_test = np.shape(features_query)[0]
    predictions = np.empty(num_test)
    for i in range(num_test):
        predictions[i] = predict_output_of_query(k, features_train, output_train, features_query[i,:])
    return predictions
        
        

sales = pd.read_csv('../Data/Week6/kc_house_data_small.csv')
train_sales = pd.read_csv('../Data/Week6/kc_house_data_small_train.csv')
test_sales = pd.read_csv('../Data/Week6/kc_house_data_small_test.csv')
validation_sales = pd.read_csv('../Data/Week6/kc_house_data_validation.csv')

features = feature_list = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view',
                           'condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated',
                           'lat','long','sqft_living15','sqft_lot15']

#Convert the above pandas dataframes to numpy arrays
(features_train,output_train) = get_numpy_data(train_sales, features, 'price')
(features_test,output_test) = get_numpy_data(test_sales, features, 'price')
(features_valid,output_valid) = get_numpy_data(validation_sales, features, 'price')

(features_train, norms) = normalize_features(features_train)
features_test = features_test/norms
features_valid = features_valid/norms

#Let the first house in the test set be query house and the 10th house in the training set be the 
#house in the training set. Compute the distances between both
#Quiz Q1:
print ('Distance between the query house and 10th house of training set ' + 
       str( np.sqrt(np.sum(np.power(np.subtract(features_test[0],features_train[9]),2))) ))

#Quiz Q2:
#Among the first ten houses, which house is closest to the query house
distances = np.sqrt(np.sum((features_train[0:10]-features_test[0])**2,axis=1))
indices_sort_order = np.argsort(distances)
print('Among the first ten houses, the house closest to query house is: ' + str(indices_sort_order[0]))

#Using vectorization of numpy, write a single-line expression to compute the Euclidean distances
#from the query point to all the instances.
distances = np.sqrt(np.sum((features_train - features_test[0])**2,axis=1))
print(distances[100]) #should print 0.232708232...

#Take the query house to be the third house in the test set
query_house = features_test[2]
dists_to_query_house = compute_distances(features_train, query_house)
indices_sort_order = np.argsort(dists_to_query_house)

#Quiz Q3
print('Index of the house closest to the query house is '+ str(indices_sort_order[0]))

#Quiz Q4
print('Predicted price of the query house based on 1NN is ' + str(output_train[indices_sort_order[0]]))

#Quiz Q5
#For the third house in the test set, find the indices of the four closest houses
four_NNs_to_query_house = k_nearest_neighbors(4, features_train, query_house)
print('The four houses closest to the query house are ' + str(four_NNs_to_query_house))
print('Predicted price of query house based on 4 NNs is ' + str(np.average(output_train[four_NNs_to_query_house])))

#Quiz Q6
#Make predictions for the first ten houses in the test set
#What is the index of the house in query set that has lowest predicted value
#What is the predicted value of that house
features_query = features_test[0:10]
predictions = predict_output(10, features_train, output_train, features_query)
req_index = np.argsort(predictions)[0]

print('The index of the house in query set that has lowest predicted value is: ' + str(req_index))
print('The predicted value of that house is: ' + str(predictions[req_index]))

#Choose the best value of k using validation set
RSS_best_k = np.Inf
best_k = 0
for k in range(1,16):
    predictions_valid = predict_output(k,features_train,output_train,features_valid)
    RSS = np.sum(np.power(np.subtract(predictions_valid,output_valid),2))
    if (RSS < RSS_best_k):
        RSS_best_k = RSS
        best_k = k  

#Quiz Q7
print('Value of k for which the RSS on validation data is minimum is ' + str(best_k))

#Quiz Q8
#Compute the RSS on the test data using the best_k computed above
predictions_test = predict_output(best_k,features_train,output_train,features_test)
RSS = np.sum(np.power(np.subtract(predictions_test,output_test),2))
print('RSS on the test_set for best_k is: ' + str(RSS))
  

