'''
Created on 28-Sep-2016
Programming Assignment 1 of Week 4 of
Regression Course of Machine Learning from Coursera
@author: sudheer
'''

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import linear_model
import matplotlib.pyplot as plt
import math

'''Given a feature and a degree, this method will return a dataframe
with the ith column equal to the ith power of the feature'''
def polynomial_dataframe(feature,degree):
    #assume that degree >=1
    #initialize the dataframe
    poly_dataframe = pd.DataFrame()
    poly_dataframe['power_1'] = feature
    #check if degree > 1
    if degree > 1:
        for power in range(2,degree+1):
            name = 'power_' + str(power)
            poly_dataframe[name] = feature.apply(lambda x: x**power)
            
    return poly_dataframe


dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('../Data/Week3/kc_house_data.csv',dtype = dtype_dict)
sales = sales.sort(['sqft_living','price'])

l2_small_penalty = 1.5e-5

#Using sqft_living as the feature, construct a dataframe with columns as 15 powers of the feature
poly15_data = polynomial_dataframe(sales['sqft_living'], 15)
model = linear_model.Ridge(alpha=l2_small_penalty,normalize=True)
model.fit(poly15_data,sales['price'])
#Answer to Quiz q1
print(model.coef_[0])

#Take 4 different data sets and try to construct models and understand their behavior
set_1 = pd.read_csv('../Data/Week3/wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
set_2 = pd.read_csv('../Data/Week3/wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
set_3 = pd.read_csv('../Data/Week3/wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
set_4 = pd.read_csv('../Data/Week3/wk3_kc_house_set_4_data.csv', dtype=dtype_dict)

set_1_poly15_data = polynomial_dataframe(set_1['sqft_living'], 15)
set_2_poly15_data = polynomial_dataframe(set_2['sqft_living'], 15)
set_3_poly15_data = polynomial_dataframe(set_3['sqft_living'], 15)
set_4_poly15_data = polynomial_dataframe(set_4['sqft_living'], 15)

l2_small_penalty = 1e-9
model_small_set = linear_model.Ridge(alpha=l2_small_penalty,normalize=True)

model_small_set.fit(set_1_poly15_data,set_1['price'])
#plt.plot(set_1['sqft_living'],set_1['price'],'.',set_1['sqft_living'],model_small_set.predict(set_1_poly15_data),'-')
#plt.show()
#print("coefficient for power_1 for dataset1 using small penalty : "+ str(model_small_set.coef_[0]))

model_small_set.fit(set_2_poly15_data,set_2['price'])
#plt.plot(set_2['sqft_living'],set_2['price'],'.',set_2['sqft_living'],model_small_set.predict(set_2_poly15_data),'-')
#plt.show()
#print("coefficient for power_1 for dataset2 using small penalty : "+ str(model_small_set.coef_[0]))

model_small_set.fit(set_3_poly15_data,set_3['price'])
#plt.plot(set_3['sqft_living'],set_3['price'],'.',set_3['sqft_living'],model_small_set.predict(set_3_poly15_data),'-')
#plt.show()
#print("coefficient for power_1 for dataset3 using small penalty : "+ str(model_small_set.coef_[0]))

model_small_set.fit(set_4_poly15_data,set_4['price'])
#plt.plot(set_4['sqft_living'],set_4['price'],'.',set_4['sqft_living'],model_small_set.predict(set_4_poly15_data),'-')
#plt.show()
#print("coefficient for power_1 for dataset4 using small penalty : "+ str(model_small_set.coef_[0]))

#Fit a 15th order polynomial on each of the above datasets, but this time
#set a large L2 Penalty
l2_large_penalty = 1.23e2
model_large_set = linear_model.Ridge(alpha=l2_large_penalty,normalize=True)

model_large_set.fit(set_1_poly15_data,set_1['price'])
#plt.plot(set_1['sqft_living'],set_1['price'],'.',set_1['sqft_living'],model_large_set.predict(set_1_poly15_data),'-')
#plt.show()
#print("coefficient for power_1 for dataset1 using large penalty : "+ str(model_large_set.coef_[0]))

model_large_set.fit(set_2_poly15_data,set_2['price'])
#plt.plot(set_2['sqft_living'],set_2['price'],'.',set_2['sqft_living'],model_large_set.predict(set_2_poly15_data),'-')
#plt.show()
#print("coefficient for power_1 for dataset2 using large penalty : "+ str(model_large_set.coef_[0]))

model_large_set.fit(set_3_poly15_data,set_3['price'])
#plt.plot(set_3['sqft_living'],set_3['price'],'.',set_3['sqft_living'],model_large_set.predict(set_3_poly15_data),'-')
#plt.show()
#print("coefficient for power_1 for dataset3 using large penalty : "+ str(model_large_set.coef_[0]))

model_large_set.fit(set_4_poly15_data,set_4['price'])
#plt.plot(set_4['sqft_living'],set_4['price'],'.',set_4['sqft_living'],model_large_set.predict(set_4_poly15_data),'-')
#plt.show()
#print("coefficient for power_1 for dataset4 using large penalty : "+ str(model_large_set.coef_[0]))

train_valid_shuffled = pd.read_csv('../Data/Week3/wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)
test = pd.read_csv('../Data/Week3/wk3_kc_house_test_data.csv', dtype=dtype_dict)

n = len(train_valid_shuffled)
k = 10 #10-fold-cross-validation

l2_penalty_choices = np.logspace(3,9,num=13)
dataframe = pd.DataFrame({'l2_penalty' : l2_penalty_choices})

def k_fold_cross_validation(k, l2_penalty, data, output):
    avg_validation_errors = []
    model = linear_model.Ridge(alpha=l2_penalty,normalize=True)
    for i in range(0,k):
        start = math.trunc((n*i)/k)
        end = math.trunc((n*(i+1))/k -1)
        validation_set_features = data[start:end+1]
        validation_set_output = output[start:end+1]
        training_set_features = data[0:start].append(data[end+1:n])
        training_set_output = output[0:start].append(output[end+1:n])
        model.fit(training_set_features,training_set_output)
        #print(model.coef_)
        avg_validation_errors.append(np.sqrt(metrics.mean_squared_error(validation_set_output,model.predict(validation_set_features))))
    return np.mean(avg_validation_errors)

input_data = polynomial_dataframe(train_valid_shuffled['sqft_living'], 15)
l2_penalty_choices = np.logspace(3,9,num=13)
l2_penalty_dataframe = pd.DataFrame({'l2_penalty' : l2_penalty_choices})
#print(k_fold_cross_validation(k, 10, input_data, pd.DataFrame( {'price':train_valid_shuffled['price']} ) ))

l2_penalty_dataframe['avg_validation_error'] = l2_penalty_dataframe.apply(lambda x: k_fold_cross_validation(k, x['l2_penalty'], input_data,train_valid_shuffled['price']),axis=1)
best_l2_penalty_value = l2_penalty_dataframe[l2_penalty_dataframe['avg_validation_error']==l2_penalty_dataframe['avg_validation_error'].min()]['l2_penalty'][0]
print ("Best L2 penalty value is: " + str(best_l2_penalty_value))

'''Now that we have got the best value for L2 penalty, we will
train the entire model with that L2 penalty value'''
best_model = linear_model.Ridge(alpha=best_l2_penalty_value, normalize=True)
best_model.fit(input_data,train_valid_shuffled['price'])
test_poly_data = polynomial_dataframe(test['sqft_living'], 15)
RSS = np.sum(np.power(np.subtract(best_model.predict(test_poly_data),test['price']),2))
print(RSS)


