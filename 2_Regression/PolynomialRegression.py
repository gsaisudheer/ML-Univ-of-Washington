'''
Created on 28-Aug-2016
Programming Assignment 2 for Week 3
of Regression course of Machine Learning
Specialization of Coursera.

Programming Assignment 1 for Week 4
@author: sudheer
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import pylab

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

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

sales = pd.read_csv('../Data/Week3/kc_house_data.csv', dtype=dtype_dict)
sales = sales.sort(['sqft_living','price'])

l2_small_penalty = 1.5e-5
#poly1_data = polynomial_dataframe(sales['sqft_living'], 1)
#poly1_data['price'] = sales['price']

poly15_data = polynomial_dataframe(sales['sqft_living'], 15)
model = linear_model.Ridge(alpha=l2_small_penalty,normalize=True)
model.fit(poly15_data, sales['price'])

##Quiz question 1
#Print the coefficient of 'feature_power1'
print (model.coef_[1])

set_1 = pd.read_csv('../Data/Week3/wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
set_2 = pd.read_csv('../Data/Week3/wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
set_3 = pd.read_csv('../Data/Week3/wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
set_4 = pd.read_csv('../Data/Week3/wk3_kc_house_set_4_data.csv', dtype=dtype_dict)

#set_1 = set_1.sort(['sqft_living','price'])
#set_2 = set_2.sort(['sqft_living','price'])
#set_3 = set_3.sort(['sqft_living','price'])
#set_4 = set_4.sort(['sqft_living','price'])

l2_small_penalty = 1e-9
#l2_small_penalty = 1.23e2    

poly15_data_set1 = polynomial_dataframe(set_1['sqft_living'], 15)
poly15_data_set2 = polynomial_dataframe(set_2['sqft_living'], 15)
poly15_data_set3 = polynomial_dataframe(set_3['sqft_living'], 15)
poly15_data_set4 = polynomial_dataframe(set_4['sqft_living'], 15)

model_1 = linear_model.Ridge(alpha=l2_small_penalty,normalize=True)
model_2 = linear_model.Ridge(alpha=l2_small_penalty,normalize=True)
model_3 = linear_model.Ridge(alpha=l2_small_penalty,normalize=True)
model_4 = linear_model.Ridge(alpha=l2_small_penalty,normalize=True)

model_1.fit(poly15_data_set1, set_1['price'])
model_2.fit(poly15_data_set2, set_2['price'])
model_3.fit(poly15_data_set3, set_3['price'])
model_4.fit(poly15_data_set4, set_4['price'])

predictions_1 = model_1.predict(poly15_data_set1)
plt.plot(set_1['sqft_living'],set_1['price'],'.',set_1['sqft_living'],predictions_1,'-')
#pylab.show()

predictions_2 = model_2.predict(poly15_data_set2)
plt.plot(set_2['sqft_living'],set_2['price'],'.',set_2['sqft_living'],predictions_2,'-')
#pylab.show()

predictions_3 = model_3.predict(poly15_data_set3)
plt.plot(set_3['sqft_living'],set_3['price'],'.',set_3['sqft_living'],predictions_3,'-')
#pylab.show()

predictions_4 = model_4.predict(poly15_data_set4)
plt.plot(set_4['sqft_living'],set_4['price'],'.',set_4['sqft_living'],predictions_4,'-')
#pylab.show()

print(model_1.coef_)
print(model_2.coef_)
print(model_3.coef_)
print(model_4.coef_)

print('minimum first coefficient is ' + str(min(model_1.coef_[1],model_2.coef_[1],model_3.coef_[1],model_4.coef_[1])))
print('maximum first coefficient is ' + str(max(model_1.coef_[1],model_2.coef_[1],model_3.coef_[1],model_4.coef_[1])))

#name = 'model1'
#A = np.zeros(4)
#A[0] = 5
#A[1] = 2
#print(A)
