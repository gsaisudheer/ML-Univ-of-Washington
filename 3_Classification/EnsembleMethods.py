'''
Created on 09-Apr-2017
Programming Assignment 1 of Week 5 from the Classification course of
Coursera Machine Learning Specialization
@author: sudheer
'''

import pandas as pd
import numpy as np
import sklearn
import sklearn.ensemble
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
import matplotlib.pyplot as plt

def turn_into_categorical_variables(data, categorical_features):
    #get one-hot encoding of the columns listed in categorical_variables
    one_hot = pd.get_dummies(data[categorical_features])
    data = data.drop(categorical_features, axis=1)
    data = data.join(one_hot)
    return data

def get_numpy_data(dataframe, features, label):
    features_frame = dataframe[features]
    features_matrix = features_frame.as_matrix()
    label_sarray = dataframe[label]
    label_array = label_sarray.as_matrix()
    return (features_matrix, label_array)

def make_figure(dim, title, xlabel, ylabel, legend):
    plt.rcParams['figure.figsize'] = dim
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend is not None:
        plt.legend(loc=legend, prop={'size':15})
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

loans = pd.read_csv('Data/Week5/lending-club-data.csv')
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x==0 else -1)
del loans['bad_loans']

target = 'safe_loans'
features = ['grade',                     # grade of the loan (categorical)
            'sub_grade_num',             # sub-grade of the loan as a number from 0 to 1
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'payment_inc_ratio',         # ratio of the monthly payment to income
            'delinq_2yrs',               # number of delinquincies
            'delinq_2yrs_zero',          # no delinquincies in last 2 years
            'inq_last_6mths',            # number of creditor inquiries in last 6 months
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'open_acc',                  # number of open credit accounts
            'pub_rec',                   # number of derogatory public records
            'pub_rec_zero',              # no derogatory public records
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
            'int_rate',                  # interest rate of the loan
            'total_rec_int',             # interest received to date
            'annual_inc',                # annual income of borrower
            'funded_amnt',               # amount committed to the loan
            'funded_amnt_inv',           # amount committed by investors for the loan
            'installment',               # monthly payment owed by the borrower
           ]

#drop NAs
loans = loans[[target] + features].dropna()
#Perform one hot encoding to convert categorical variables into binary features
loans_categorical = turn_into_categorical_variables(loans, features)

train_idx = pd.read_json('Data/Week5/train-idx.json').values[:,0] #indices of the training data
valid_idx = pd.read_json('Data/Week5/validation-idx.json').values[:,0] #indices of the test data

#Construct the train and validation data
train_data = loans_categorical.iloc[train_idx]
validation_data = loans_categorical.iloc[valid_idx]

categorical_features = train_data.columns.values
categorical_features = categorical_features[categorical_features != target]

#convert to numpy matrix
train_data_features = train_data[categorical_features].as_matrix()
train_data_output = train_data[target].as_matrix()

#train an ensemble of 5 trees
model_5 = GradientBoostingClassifier(max_depth=6, n_estimators=5)
model_5.fit(train_data_features,train_data_output)

validation_safe_loans = validation_data[validation_data[target] == +1]
validation_risky_loans = validation_data[validation_data[target] == -1]

#Predict on a lower sample validation data
sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
predictions = model_5.predict(sample_validation_data[categorical_features].as_matrix())
correct_predictions_percentage = np.sum(np.equal(sample_validation_data[target].as_matrix(), predictions))/np.shape(predictions)[0]
print('model_5 got %s of predictions on sample_validation_data right' %correct_predictions_percentage)

probabilities_safe = model_5.predict_proba(sample_validation_data[categorical_features].as_matrix())[:,1]
print(probabilities_safe)
print('The index (1-based) of loan that has lowest probability of being classified as safe is %s' %(np.argmin(probabilities_safe)+1))

#Evaluate the accuracy of model_5 on the validation data
print('Accuracy of model_5 on the validation data is', model_5.score(validation_data[categorical_features].as_matrix(),
                                                                     validation_data[target].as_matrix()))

#Compute the number of false positives and false negatives made by model_5 on the validation data
compare_output = pd.DataFrame({'Actual':pd.Series(validation_data[target].as_matrix()), 
                               'Predicted': pd.Series(model_5.predict(validation_data[categorical_features].as_matrix()))})

false_positives = np.sum(compare_output.apply(lambda x: x['Actual'] == -1 and x['Predicted'] == +1, axis=1))
false_negatives = np.sum(compare_output.apply(lambda x: x['Actual'] == +1 and x['Predicted'] == -1, axis=1))
print('On validation data, Number of false positives is %s and number of false negatives is %s' %(false_positives, false_negatives))

#Assuming that each false_positive costs 20K and each false_negative costs 10K, find the total cost made by model_5
print('Cost incurred by model_5 is %s' %(10000 * false_negatives + 20000 * false_positives))

#Using model_5, make probability predictions for all the points in the validation data
#Add these into validation_data as a column called predictions
validation_data.loc[:,'predictions'] = model_5.predict_proba(validation_data[categorical_features].as_matrix())[:,1]

#Sort the data in validation_data in decreasing order of 'predictions'
validation_data_sorted = validation_data.sort_values(by = ['predictions'],ascending = [False])
print('Printing the values of validation_data_sorted')
print(validation_data_sorted.iloc[0:2])
print('Printing the values of validation_data')
print(validation_data.iloc[0:2])
filter_col = [col for col in list(validation_data_sorted) if col.startswith('grade')]
print('Printing the grades of the top 5 safe loans')
print(validation_data_sorted[filter_col].iloc[0:5])
print('==========================================')

print('Printing the grades of the top 5 risky loans')
print(validation_data_sorted[filter_col].iloc[-5:])
print('==========================================')

validation_data_features = validation_data[categorical_features].as_matrix()
validation_data_target = validation_data[target].as_matrix()

numTrees = [10,50,100,200,500]
training_errors =[]
validation_errors=[]

for trees in numTrees:
    model_x = None
    model_x = GradientBoostingClassifier(max_depth=6, n_estimators=trees)
    model_x.fit(train_data_features,train_data_output)
    training_errors.append(1.0-model_x.score(train_data_features,train_data_output))
    validation_errors.append(1.0-model_x.score(validation_data_features,validation_data_target))


print('Printing training errors')
print(training_errors)
print('Printing validation errors')
print(validation_errors)
plt.plot([10, 50, 100, 200, 500], training_errors, linewidth=4.0, label='Training error')
plt.plot([10, 50, 100, 200, 500], validation_errors, linewidth=4.0, label='Validation error')

make_figure(dim=(10,5), title='Error vs number of trees',
            xlabel='Number of trees',
            ylabel='Classification error',
            legend='best')
plt.show()




