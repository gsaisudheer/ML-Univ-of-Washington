'''
Created on 12-Mar-2017
Programming Assignment 1 of Week 3 from the Classification course of
Coursera Machine Learning Specialization
@author: sudheer
'''

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

def identify_categorical_variables(data):
    categorical = []
    for feat_name, feat_type in zip(data.columns, data.dtypes):
        if feat_type == object:
            categorical.append(feat_name)
    return categorical

def turn_into_categorical_variables(data, categorical_features):
    #get one-hot encoding of the columns listed in categorical_variables
    one_hot = pd.get_dummies(data[categorical_features])
    data = data.drop(categorical_features, axis=1)
    data = data.join(one_hot)
    return data
    
    
loans = pd.read_csv('/Users/sudheer/Coursera/Machine Learning Specialization/3_Classification/MLClassification/Data/Week3/3a/lending-club-data.csv')
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x==0 else -1)
del loans['bad_loans']

#Work on only the following features 
features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'                    # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
loans = loans[features + [target]]
#loans['term'] = loans['term'].apply(lambda x: float(re.sub('[^0-9]','', x)))

#Load the indices of training and validation data
train_idx = pd.read_json('/Users/sudheer/Coursera/Machine Learning Specialization/3_Classification/MLClassification/Data/Week3/3a/module-5-assignment-1-train-idx.json').values[:,0]
valid_idx = pd.read_json('/Users/sudheer/Coursera/Machine Learning Specialization/3_Classification/MLClassification/Data/Week3/3a/module-5-assignment-1-validation-idx.json').values[:,0]

train_data = loans.iloc[train_idx]
validation_data = loans.iloc[valid_idx]

#print(train_data.dtypes)
#get numpy data of the features and the target
#loans_features_train, loans_target_train = get_numpy_data(train_data, features, target)
#loans_features_train = train_data[features].as_matrix()
#loans_target_train = train_data[features].as_matrix()

#identify categorical features
categorical_types = identify_categorical_variables(train_data)
train_data_categorical = turn_into_categorical_variables(train_data, categorical_types)
#train_data_categorical.to_csv('/Users/sudheer/Coursera/Machine Learning Specialization/3_Classification/MLClassification/Data/Week3/3a/train_data_from_pandas.csv')
print(train_data_categorical.iloc[0:2])
train_data_features = pd.DataFrame.copy(train_data_categorical)
del train_data_features[target]
train_data_target = train_data_categorical[target]

#construct a decision tree classifier
decision_tree_model = DecisionTreeClassifier(max_depth=6)
decision_tree_model.fit(train_data_features,train_data_target)
small_model = DecisionTreeClassifier(max_depth=2)
small_model.fit(train_data_features,train_data_target)
export_graphviz(small_model, out_file='/Users/sudheer/Desktop/tree.dot')

#Grab 2 positive and 2 negative examples
validation_data_categorical = turn_into_categorical_variables(validation_data,categorical_types)
validation_safe_loans = validation_data_categorical[validation_data_categorical[target] == 1]
validation_risky_loans = validation_data_categorical[validation_data_categorical[target] == -1]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
print('Printing sample validation data')
print(sample_validation_data)

#For each row in sample_validation_data, use decision_tree_model to predict the outcome
sample_validation_data_predictions = decision_tree_model.predict(sample_validation_data.ix[:,sample_validation_data.columns != target])
print('printing sample validation_data predictions')
print(sample_validation_data_predictions)

#Quiz Q1:
#What percentage of sample_validation_data did decision_tree_model get correct?
correctly_classified = np.sum(np.equal(sample_validation_data_predictions,sample_validation_data[target]))
accuracy = correctly_classified/np.shape(sample_validation_data_predictions)
print('accuracy of decision_tree model on validation data is ' + str(accuracy))

#Quiz Q2:
#Which loan has the highest probability of being classified as safe loan by decision_tree_model?
sample_validation_data_prob = decision_tree_model.predict_proba(sample_validation_data.ix[:,sample_validation_data.columns != target])
sample_validation_data_prob = sample_validation_data_prob[:,1]
print('loan that is higly probable of being safe is ' + str(np.argmax(sample_validation_data_prob)+1))

#Evaluate the small_model and decision_tree_model on the training data
print('score of small_model on training data is ' + str(small_model.score(train_data_features, train_data_target)))
print('score of decision_tree_model on training data is ' + str(decision_tree_model.score(train_data_features,train_data_target)))

#Evaluate the small_model and decision_tree_model on the entire validation data
validation_data_features = validation_data_categorical.ix[:,validation_data_categorical.columns != target]
validation_data_target = validation_data_categorical[target]
print('score of small_model on validation data is ' + str(small_model.score(validation_data_features,validation_data_target)))
print('score of decision_tree_model on validation data is ' + str(decision_tree_model.score(validation_data_features,validation_data_target)))


#Build a complex model with depth 10
big_model = DecisionTreeClassifier(max_depth=10)
big_model.fit(train_data_features,train_data_target)
print('score of big_model on training data is ' + str(big_model.score(train_data_features,train_data_target)))
print('score of big_model on validation data is ' + str(big_model.score(validation_data_features,validation_data_target)))

#Compute the number of mistakes on validation data made by decision_tree_model
decision_tree_model_predictions = decision_tree_model.predict(validation_data_features)
observation_table = pd.DataFrame({'Actual':validation_data_target,'Predicted':decision_tree_model_predictions})
#compute false negatives,i.e. loans that were actually safe but predicted to be risky
false_negatives = np.sum(observation_table.apply(lambda x: x['Actual'] == +1 and x['Predicted'] == -1, axis=1))
#compute false positives,i.e. loans that were actually risky but predicted to be safe
false_positives = np.sum(observation_table.apply(lambda x: x['Actual'] == -1 and x['Predicted'] == +1, axis=1))
#if each false negative costs 10K and each false positive costs 20K, compute the total cost
cost_incurred = false_positives*20000 + false_negatives * 10000
print('cost incurred for the mistakes is ' + str(cost_incurred))

