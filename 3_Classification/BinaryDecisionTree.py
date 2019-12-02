'''
Created on 08-Apr-2017
Programming Assignment 2 of Week 3 from the Classification course of
Coursera Machine Learning Specialization
@author: sudheer
'''

import pandas as pd
import  numpy as np

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

def intermediate_node_num_mistakes(labels_in_node):
    if len(labels_in_node) == 0:
        return 0
    #Count the number of 1's (safe loans)
    positive_loans = (labels_in_node.where(labels_in_node == +1)).count()
    #Count the number of -1's (risky loans)
    negative_loans = (labels_in_node.where(labels_in_node == -1)).count()
    #Return the number of mistakes. Since, we are using majority class, points that are
    #not in majority class are considered as mistakes
    if positive_loans > negative_loans: #majority class prediction is positive. 
        return negative_loans #num mistakes is number of negative loans
    else:
        return positive_loans
    
def best_splitting_feature(data, features, target):
    target_values = data[target]
    best_feature = None #Keep track of best feature
    best_error = 10  #Keep track of best error so far
    
    #Convert to float to make sure that error gets computed correctly
    num_data_points = float(len(data.index))
    
    #Loop through each feature for considering to split on that feature
    for feature in features:
        #Left split will have all data points where feature value is 0
        left_split = data[data[feature] == 0]
        
        #Right split will have all data points where feature value is 1
        right_split = data[data[feature] == 1]
        
        #Calculate the number of misclassified examples in the left node
        left_mistakes = intermediate_node_num_mistakes(left_split[target])
        
        #Calculate the number of misclassified examples in the right node
        right_mistakes = intermediate_node_num_mistakes(right_split[target])
        
        #Compute the classification error of this split
        error = (left_mistakes + right_mistakes)/num_data_points
        
        #if error is less than best_error, store feature as best_feature and error as best_error
        if error < best_error:
            best_feature = feature
            best_error = error
    
    return best_feature 

def create_leaf (target_values):
    #Create a leaf node
    leaf = {'splitting_feature': None,
            'left': None,
            'right': None,
            'is_leaf': True}
    #Count the number of nodes that are +1 and -1 in this node
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])
    #For the leaf node, set the prediction to be the majority class
    if num_ones > num_minus_ones:
        leaf['prediction'] = +1
    else:
        leaf['prediction'] = -1
    #Return leaf node
    return leaf 

def decision_tree_create(data, features, target, current_depth = 0, max_depth = 10):
    remaining_features = features[:]
    target_values = data[target]
    print ("--------------------------------------------------------------------")
    print ("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))
    
    #Stopping condition 1
    #Check if there are mistakes at current node
    mistakes = intermediate_node_num_mistakes(target_values)
    if mistakes == 0:
        print ('Stopping condition 1 reached.')
        #if no mistakes at current node, make current node a leaf node
        return create_leaf(target_values)
    
    #Stopping condition 2
    #Check if there are no more features to split on
    if remaining_features == None:
        print('Stopping condition 2 reached.')
        #if there are no more remaining features to consider, make it a leaf node
        return create_leaf(target_values)
    
    #Stopping condition 3 (limit tree depth)
    if current_depth >= max_depth:
        print('Reached maximum depth. Stopping for now')
        #if max tree depth is reached, make current node a leaf node
        return create_leaf(target_values)
    
    #Find the best splitting feature
    splitting_feature = best_splitting_feature(data, remaining_features, target)
    
    #Split on the best feature that was found
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    remaining_features = remaining_features[remaining_features != splitting_feature]
    print("Split on feature %s. (%s, %s)" %(\
                                            splitting_feature, len(left_split), len(right_split)))
    
    #Create a leaf node if the split is perfect
    if len(left_split) == len(data):
        print('Creating a leaf node')
        return create_leaf(left_split[target])
    if len(right_split) == len(data):
        print('Creating a leaf node')
        return create_leaf(right_split[target])
    
    #Repeat on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth)
    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth)
    
    return{'is_leaf': False,
           'prediction': None,
           'splitting_feature': splitting_feature,
           'left': left_tree,
           'right': right_tree}

def classify(tree, x, annotate = False):
    #if the tree is a leaf node.
    if tree['is_leaf']:
        if annotate:
            print("At leaf, predicting %s" % tree['prediction'])
        return tree['prediction']
    else:
        #split on feature
        split_feature_value = x[tree['splitting_feature']]
        if annotate:
            print("Split on %s = %s" % (tree['splitting_feature'], split_feature_value))
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)

def evaluate_classification_error(tree, data, target):
    #Apply the classify(tree,x) for each x in the data
    prediction = data.apply(lambda x: classify(tree,x), axis=1)
    actual = data[target]
    return 1-(np.sum(np.equal(prediction,actual))*1.0/np.shape(actual)[0])

def print_stump(tree, name = 'root'):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print ("(leaf, label: %s)" % tree['prediction'])
        return None
    #split_feature, split_value = split_name.split('.')
    print ('                       %s' % name)
    print ('         |---------------|----------------|')
    print ('         |                                |')
    print ('         |                                |')
    print ('         |                                |')
    print ('  [{0} == 0]               [{0} == 1]    '.format(split_name))
    print ('         |                                |')
    print ('         |                                |')
    print ('         |                                |')
    print ('    (%s)                         (%s)' \
        % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))) 
                 
loans = pd.read_csv('Data/Week3/3a/lending-club-data.csv')
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x==0 else -1)
del loans['bad_loans']

#We will be considering only the following features
features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'

#Extract from the loans data only the above features and the target
loans = loans[features+[target]]
train_idx = pd.read_json('Data/Week3/3b/train-idx.json').values[:,0] #indices of the training data
test_idx = pd.read_json('Data/Week3/3b/test-idx.json').values[:,0] #indices of the test data

#Form training and test data
train_data = loans.iloc[train_idx]
test_data = loans.iloc[test_idx]

#Transform train_data and test_data such that they have categorical values for the features
train_data_categorical = turn_into_categorical_variables(train_data, features)
test_data_categorical = turn_into_categorical_variables(test_data, features)

categorical_features = train_data_categorical.columns.values
categorical_features = categorical_features[categorical_features != target]
#Train a tree model with max_depth = 6
my_decision_tree = decision_tree_create(train_data_categorical,categorical_features, target, 0, 6)

print(test_data_categorical.iloc[0])
print("Predicted class: %s " % classify(my_decision_tree, test_data_categorical.iloc[0][categorical_features]))

#Find the prediction path to find the output for the first test point
classify(my_decision_tree, test_data_categorical.iloc[0],True)

#Evaluate the classification error on the test_data
classification_error = evaluate_classification_error(my_decision_tree, test_data_categorical, target)
print('classification_error is %s' %classification_error)
print_stump(my_decision_tree)

print_stump(my_decision_tree['right']['right'], my_decision_tree['right']['splitting_feature'])






