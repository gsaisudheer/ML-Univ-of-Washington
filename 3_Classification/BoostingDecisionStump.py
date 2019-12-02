'''
Created on 11-Apr-2017
Programming Assignment 2 of Week 5 from the Classification course of 
Machine Learning Specialization
@author: sudheer
'''

import pandas as pd
import numpy as np
from math import log
from math import exp
import matplotlib.pyplot as plt

def turn_into_categorical_variables(data, categorical_features):
    #get one-hot encoding of the columns listed in categorical_variables
    one_hot = pd.get_dummies(data[categorical_features])
    data = data.drop(categorical_features, axis=1)
    data = data.join(one_hot)
    return data

def intermediate_node_weighted_mistakes(labels_in_node, data_weights):
    #Sum the weights of all entries with labels +1
    total_weight_positive = np.sum(data_weights[(labels_in_node == +1).values])
    #Weight of mistakes for predicting all -1s is equal to the sum above
    weighted_mistakes_all_negative = total_weight_positive   
    #Sum the weights of all entries with labels -1
    total_weight_negative = np.sum(data_weights[(labels_in_node == -1).values])
    #Weight of mistakes for predicting all +1s is equal to the sum above
    weighted_mistakes_all_positive = total_weight_negative
    #Return the tuple (weight, class_label) representing the lower of the two weights
    if weighted_mistakes_all_negative < weighted_mistakes_all_positive:
        return (weighted_mistakes_all_negative, -1)
    return (weighted_mistakes_all_positive, +1)

def best_splitting_feature(data, features, target, data_weights):
    #If data is identical in each feature, this function should return None
    best_feature = None
    best_error = float('+inf')
    
    #Loop through each feature to consider splitting on that feature
    for feature in features:
        #Left split will have data points where feature value is 0 
        #Right split will have data points where feature value is 1
        left_split = data[data[feature] == 0]
        right_split = data[data[feature] == 1]
        
        #Apply same filtering to create left data weights and right data weights
        #Need to be done carefully as data_weights is numpy array and data is pd.DataFrame
        left_data_weights = data_weights[(data[feature] == 0).values]
        right_data_weights = data_weights[(data[feature] == 1).values]
        
        '''
        print('left_data_weights_sum: %s' %np.sum(left_data_weights))
        print('right_data_weights_sum: %s' %np.sum(right_data_weights))
        print('data_weights_sum: %s' %np.sum(data_weights))'''
        
        #Calculate the weights of mistakes for left and right sides
        left_weighted_mistakes, left_class = intermediate_node_weighted_mistakes(left_split[target], left_data_weights)
        right_weighted_mistakes, right_class = intermediate_node_weighted_mistakes(right_split[target], right_data_weights)
        
        error = (left_weighted_mistakes + right_weighted_mistakes)/(np.sum(left_data_weights) + np.sum(right_data_weights))
        
        if error < best_error:
            best_error = error
            best_feature = feature
    
    #Return the best feature that we found
    return best_feature

def create_leaf(target_values, data_weights):
    #Create a leaf Node
    leaf = {'splitting_feature': None,
            'is_leaf': True}
    weighted_error, best_class = intermediate_node_weighted_mistakes(target_values, data_weights)
    leaf['prediction'] = best_class
    return leaf

def weighted_decision_tree_create(data, features, target, data_weights, current_depth =1, max_depth = 10):
    remaining_features = features[:] #Make a copy of features
    target_values = data[target]
    print ("--------------------------------------------------------------------")
    print ("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))
    
    #Stopping condition 1. Error is 0
    if intermediate_node_weighted_mistakes(target_values, data_weights)[0] <= 1e-15:
        print('Stopping condition 1 reached')
        return create_leaf(target_values,data_weights)
    
    #Stopping condition 2. No more features
    if remaining_features == []:
        print('Stopping condition 2 reached.')
        return create_leaf(target_values, data_weights)
    
    #Additional stopping condition. Limit tree depth
    if current_depth > max_depth:
        print('Reached maximum depth. Stopping for now')
        return create_leaf(target_values,data_weights)
    
    splitting_feature = best_splitting_feature(data, features, target, data_weights)
    remaining_features = remaining_features[remaining_features != splitting_feature]
    
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    
    left_data_weights = data_weights[data[splitting_feature] == 0]
    right_data_weights = data_weights[data[splitting_feature] == 1]
    
    print ("Split on feature %s. (%s, %s)" % (\
              splitting_feature, len(left_split), len(right_split)))
    
    #Create a leaf node if the split is perfect
    if(len(left_split) == len(data)):
        print('Creating leaf node')
        return create_leaf(left_split[target], left_data_weights)
    if(len(right_split) == len(data)):
        print('Creating leaf node')
        return create_leaf(right_split[target], right_data_weights) 
    
    #Recurse on left and right subtrees
    left_tree = weighted_decision_tree_create(left_split, remaining_features, target, left_data_weights, current_depth+1, max_depth)
    right_tree = weighted_decision_tree_create(right_split, remaining_features, target, right_data_weights, current_depth+1, max_depth)
    
    return {'is_leaf': False,
            'prediction': None,
            'splitting_feature':splitting_feature,
            'left': left_tree,
            'right': right_tree}

def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])

def classify(tree, x, annotate=False):
    #If the node is a leaf node.
    if tree['is_leaf']:
        if annotate:
            print('At leaf, predicting %s' %tree['prediction'])
        return tree['prediction']
    else:
        #Split on feature
        split_feature_value = x[tree['splitting_feature']]
        if annotate:
            print('Split on %s = %s '% (tree['splitting_feature'], split_feature_value))
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)

def evaluate_classification_error(tree, data, target):
    #Apply the classify(tree,x) to each row in the data
    prediction = data.apply(lambda x: classify(tree, x), axis=1)
    
    #Once the predictions are done, calculate the classification error
    return (prediction!= data[target]).sum()/float(len(data))

def adaboost_with_tree_stumps(data, features, target, num_tree_stumps):
    #start with unweighted data
    alpha = np.asarray([1.]*len(data))
    weights = []
    tree_stumps = []
    target_values = data[target]
    
    for t in range(num_tree_stumps):
        print('=============================================')
        print('Adaboost iteration %d' %t)
        print('=============================================')
        #Learn a weighted decision tree stump. Use max_depth = 1
        tree_stump = weighted_decision_tree_create(data, features, target, data_weights=alpha, current_depth=1, max_depth=1)
        tree_stumps.append(tree_stump)
        
        #Make predictions 
        predictions = data.apply(lambda x: classify(tree_stump, x), axis=1)
        is_correct = predictions == target_values
        is_wrong = predictions != target_values
        
        #Compute weighted error
        weighted_error = alpha[is_wrong.values]
        weighted_error = np.sum(weighted_error)/np.sum(alpha)
        
        #Compute model coefficient using weighted error
        weight = 0.5 * log((1-np.sum(weighted_error))/np.sum(weighted_error))
        weights.append(weight)
        
        #Adjust weights on data point
        adjustment = is_correct.apply(lambda correct: exp(-weight) if correct else exp(weight))
        
        #Scale alpha by multiplying the adjustment
        #Then normalize data weights
        alpha = np.multiply(alpha, adjustment)
        alpha = np.divide(alpha, np.sum(alpha))
        
    return weights, tree_stumps

def predict_adaboost(stump_weights, tree_stumps, data):
    scores = np.asarray([0.]*len(data))
    
    for i, tree in enumerate(tree_stumps):
        predictions = data.apply(lambda x: classify(tree, x), axis=1)  
        #Accumulate predictions on the scores array
        scores = np.add(scores, stump_weights[i]*predictions)
        
    return scores.apply(lambda score: +1 if score > 0 else -1)

            
          
loans = pd.read_csv('Data/Week5/5b/lending-club-data.csv')
features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
del loans['bad_loans']
target = 'safe_loans'
loans = loans[features + [target]]

train_idx = pd.read_json('Data/Week5/5b/train-idx.json').values[:,0] #indices of the training data
valid_idx = pd.read_json('Data/Week5/5b/test-idx.json').values[:,0] #indices of the test data

loans_categorical = turn_into_categorical_variables(loans, features)
categorical_features = loans_categorical.columns.values
categorical_features = categorical_features[categorical_features != target]

train_data = loans_categorical.iloc[train_idx]
valid_data = loans_categorical.iloc[valid_idx]

train_data_features = train_data[categorical_features]
train_data_target = train_data[target]

valid_data_features = valid_data[categorical_features]
valid_data_target = valid_data[target]


example_data_weights = np.asarray(len(train_data)*[1.5])
print(train_data.columns.values)
print(best_splitting_feature(train_data, categorical_features, target, example_data_weights))


#Create an example with first 10 and last 10 having weights 1.0 and other data points with weights 0.
example_data_weights = np.asarray([1.]*10 + [0.]*(len(train_data) - 20) + [1.]*10)
#Train a weighted_decision_tree model
small_data_decision_tree_subset_20 = weighted_decision_tree_create(train_data, categorical_features, target, example_data_weights, max_depth=2)
#Evaluate the classification error on first 10 and last 10 points vs the whole train_data
subset_20 = train_data.iloc[0:10].append(train_data.iloc[-10:])
print('Classification error on the training data by small_decision_tree_subset_20 is %s' 
      %evaluate_classification_error(small_data_decision_tree_subset_20, train_data, target))
print('Classification error on the subset_20 data by small_decision_tree_subset_20 is %s'
      %evaluate_classification_error(small_data_decision_tree_subset_20, subset_20, target))
#print(subset_20.iloc[:])
predictions = subset_20.apply(lambda x: classify(small_data_decision_tree_subset_20,x,True), axis=1)


data_weights = np.asarray([1.]*len(train_data))
stump_weights, tree_stumps = adaboost_with_tree_stumps(train_data, categorical_features, target, 30)

#Computing training error at the end of each iteration
error_all = []
for n in range(1,31):
    predictions = predict_adaboost(stump_weights[:n], tree_stumps[:n], train_data)
    mistakes = predictions != train_data[target]
    error = (1.0 *np.sum(mistakes))/len(train_data)
    error_all.append(error)
    print("Iteration %s, training error = %s" %(n,error_all[n-1]))

#Computing test error at the end of each iteration
test_error_all = []
for n in range(1,31):
    predictions = predict_adaboost(stump_weights[:n], tree_stumps[:n], valid_data)
    mistakes = predictions != valid_data[target]
    error = (1.0 *np.sum(mistakes))/len(valid_data)
    test_error_all.append(error)
    print("Iteration %s, test error = %s" %(n,test_error_all[n-1]))
    
#Visualizing training error vs the number of iterations
plt.rcParams['figure.figsize'] = 7, 5
plt.plot(range(1,31), error_all, '-', linewidth=4.0, label='Training error')
plt.plot(range(1,31), test_error_all, '-', linewidth=4.0, label='Test error')

plt.title('Performance of Adaboost ensemble')
plt.xlabel('# of iterations')
plt.ylabel('Classification error')
plt.rcParams.update({'font.size': 16})
plt.legend(loc='best', prop={'size':15})
plt.tight_layout()
plt.show()


'''
train_data = train_data.iloc[0:10]
predictions = np.asarray([-1] * 3 + [1]*4 + [-1]*3)
target_values = train_data[target]
is_correct = predictions == target_values
print('Printing is_correct')
print(is_correct)
example_weights = np.asarray([0.] * 2 + [1.] * 4 + [0.]*2 + [1.]*2)
print('Printing example_weights')
print(example_weights)
print(example_weights[is_correct.values])
print(np.sum(example_weights[is_correct.values]))

print(np.sum(example_data_weights[example_data_weights == 1]))
print('printing example_data_weights')
print(example_data_weights)
print('printing example')
print(example)
print(np.sum(example[example_data_weights == 0]))
'''
