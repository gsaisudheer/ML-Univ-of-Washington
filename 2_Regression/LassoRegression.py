'''
Created on 05-Nov-2016
Programming Assignment 1 of Week4 of 
Regression Course of Machine Learning from Coursera
@author: sudheer
'''

import pandas as pd
import numpy as np
from math import log, sqrt
from sklearn import linear_model

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('../Data/Week5/kc_house_data.csv', dtype = dtype_dict)

#Add additional features
sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']
sales['floors_square'] = sales['floors']*sales['floors']

#Use all features for constructing the lass model
all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

model_all = linear_model.Lasso(alpha=5e2, normalize=True) # set parameters
model_all.fit(sales[all_features], sales['price']) # learn weights

print(model_all.coef_)

#To find a good L1 penalty, we'll explore multiple values using a validation set.
testing = pd.read_csv('../Data/Week5/wk3_kc_house_test_data.csv', dtype=dtype_dict)
training = pd.read_csv('../Data/Week5/wk3_kc_house_train_data.csv', dtype=dtype_dict)
validation = pd.read_csv('../Data/Week5/wk3_kc_house_valid_data.csv', dtype=dtype_dict)

#Make sure to add the new features to these datasets.
testing['sqft_living_sqrt'] = testing['sqft_living'].apply(sqrt)
testing['sqft_lot_sqrt'] = testing['sqft_lot'].apply(sqrt)
testing['bedrooms_square'] = testing['bedrooms']*testing['bedrooms']
testing['floors_square'] = testing['floors']*testing['floors']

training['sqft_living_sqrt'] = training['sqft_living'].apply(sqrt)
training['sqft_lot_sqrt'] = training['sqft_lot'].apply(sqrt)
training['bedrooms_square'] = training['bedrooms']*training['bedrooms']
training['floors_square'] = training['floors']*training['floors']

validation['sqft_living_sqrt'] = validation['sqft_living'].apply(sqrt)
validation['sqft_lot_sqrt'] = validation['sqft_lot'].apply(sqrt)
validation['bedrooms_square'] = validation['bedrooms']*validation['bedrooms']
validation['floors_square'] = validation['floors']*validation['floors']

l1_penalty_choices = np.logspace(1,7,num=13)
l1_penalty_dataframe = pd.DataFrame({'l1_penalty':l1_penalty_choices})

def get_validation_error(l1_penalty,training,validation):
    model = linear_model.Lasso(alpha=l1_penalty,normalize=True)
    model.fit(training[all_features],training['price'])
    predictions = model.predict(validation[all_features])
    return np.sum(np.power(np.subtract(predictions,validation['price']),2))

l1_penalty_dataframe['validation_error'] = l1_penalty_dataframe.apply(lambda x: get_validation_error(x['l1_penalty'], training, validation), axis=1)
best_l1_penalty = l1_penalty_dataframe[l1_penalty_dataframe['validation_error']==l1_penalty_dataframe['validation_error'].min()]['l1_penalty'][0]

print('best_l1_penalty ' + str(best_l1_penalty))
#Compute the RSS on the test data for the chosen l1 penalty
best_model = linear_model.Lasso(alpha=10,normalize=True)
best_model.fit(training[all_features],training['price'])
predictions_test_data = best_model.predict(testing[all_features])

#Count the number of non-zero coefficients chosen by the best l1 penalty value
num_nonzeros_lasso = np.count_nonzero(best_model.coef_) + np.count_nonzero(best_model.intercept_)
print(num_nonzeros_lasso)

max_nonzeros = 7
wide_range_l1_penalies = np.logspace(1,4,num=20)
wide_range_l1_penalties_df = pd.DataFrame({'l1_penalty':wide_range_l1_penalies})

def get_num_nonzeros(l1_penalty,training,validation):
    model = linear_model.Lasso(alpha=l1_penalty,normalize=True)
    model.fit(training[all_features],training['price'])
    return np.count_nonzero(model.coef_)+np.count_nonzero(model.intercept_)

wide_range_l1_penalties_df['num_nonzeros'] = wide_range_l1_penalties_df.apply(lambda x: get_num_nonzeros(x['l1_penalty'], training, validation), axis=1)

#Find the largest l1_penalty that has more non-zeros than 'max_nonzeros'

temp_list = wide_range_l1_penalties_df[wide_range_l1_penalties_df['num_nonzeros']>max_nonzeros]
l1_penalty_min = temp_list[temp_list['l1_penalty']==temp_list['l1_penalty'].max()]['l1_penalty'].values[0]
print('l1_penalty_min: '+ str(l1_penalty_min))

#Find the smallest l1_penalty that has fewer non-zeros than 'max_nonzeros'
temp_list = wide_range_l1_penalties_df[wide_range_l1_penalties_df['num_nonzeros']<max_nonzeros]
l1_penalty_max = temp_list[temp_list['l1_penalty']==temp_list['l1_penalty'].min()]['l1_penalty'].values[0]
print('l1_penalty_max: ' + str(l1_penalty_max))

narrow_range_l1_penalties = np.where(np.logical_and(wide_range_l1_penalies >= l1_penalty_min, wide_range_l1_penalies <= l1_penalty_max))
narrow_range_l1_penalties = wide_range_l1_penalies[narrow_range_l1_penalties]

narrow_range_l1_penalties_df = pd.DataFrame({'l1_penalty':narrow_range_l1_penalties})
narrow_range_l1_penalties_df['validation_error'] = narrow_range_l1_penalties_df.apply(lambda x: get_validation_error(x['l1_penalty'], training, validation), axis=1)
narrow_range_l1_penalties_df['num_nonzeros'] = narrow_range_l1_penalties_df.apply(lambda x: get_num_nonzeros(x['l1_penalty'], training, validation), axis=1)
narrow_range_l1_penalties_df = narrow_range_l1_penalties_df[narrow_range_l1_penalties_df['num_nonzeros']==max_nonzeros]

best_l1_penalty = narrow_range_l1_penalties_df[narrow_range_l1_penalties_df['validation_error']==narrow_range_l1_penalties_df['validation_error'].min()]['l1_penalty'].values[0]
print('best_l1_penalty '+str(best_l1_penalty))
#Find the features selected by this best_l1_penalty
model = linear_model.Lasso(alpha=best_l1_penalty,normalize=True)
model.fit(training[all_features],training['price'])
print(model.coef_)