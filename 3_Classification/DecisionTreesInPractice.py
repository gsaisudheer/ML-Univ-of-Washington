'''
Created on 08-Apr-2017
Programming Assignment of Week 4 from the Classification Course of
Coursera Machine Learning Specialization
@author: sudheer
'''

import pandas as pd
import numpy as np

def turn_into_categorical_variables(data, categorical_features):
    #get one-hot encoding of the columns listed in categorical_variables
    one_hot = pd.get_dummies(data[categorical_features])
    data = data.drop(categorical_features, axis=1)
    data = data.join(one_hot)
    return data

def reached_minimum_node_size(data, min_node_size):
    #Return if number of data points in the node is less than or 
    #equal to min_node_size
    if len(data.index) <= min_node_size:
        return True
    return False

def error_reduction(error_before_split, error_after_split):
    #Return the error_before_split minus error_after_split
    return error_before_split - error_after_split

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

def decision_tree_create(data, features, target, current_depth = 0, max_depth = 10, min_node_size=1, min_error_reduction=0.0):
    remaining_features = features[:]
    target_values = data[target]
    '''
    print ("--------------------------------------------------------------------")
    print ("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))
    '''
    #Stopping condition 1
    #Check if there are mistakes at current node
    mistakes = intermediate_node_num_mistakes(target_values)
    if mistakes == 0:
        #print ('Stopping condition 1 reached.')
        #if no mistakes at current node, make current node a leaf node
        return create_leaf(target_values)
    
    #Stopping condition 2
    #Check if there are no more features to split on
    if remaining_features == None:
        #print('Stopping condition 2 reached.')
        #if there are no more remaining features to consider, make it a leaf node
        return create_leaf(target_values)
    
    #Early stopping condition 1 (limit tree depth)
    if current_depth >= max_depth:
        #print('Early stopping condition 1 reached. Reached maximum depth')
        #if max tree depth is reached, make current node a leaf node
        return create_leaf(target_values)
    
    #Early Stopping Condition 2 (Reached the minimum node size)
    if reached_minimum_node_size(data, min_node_size):
        #print('Early stopping condition 2 reached. Reached minimum node size')
        return create_leaf(target_values)
    
    #Find the best splitting feature
    splitting_feature = best_splitting_feature(data, remaining_features, target)
    
    #Split on the best feature that was found
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    
    #Early stopping condition 3 (Minimum error reduction)
    error_before_split = intermediate_node_num_mistakes(target_values)/float(len(data.index))
    
    left_mistakes = intermediate_node_num_mistakes(left_split[target])
    right_mistakes = intermediate_node_num_mistakes(right_split[target])
    error_after_split = (left_mistakes + right_mistakes)/float(len(data.index))
    
    #If the error reduction is less than or equal to min_error_reduction, return a leaf
    if error_reduction(error_before_split, error_after_split) <= min_error_reduction:
        return create_leaf(target_values)
    
    remaining_features = remaining_features[remaining_features != splitting_feature]
    '''
    print("Split on feature %s. (%s, %s)" %(\
                                            splitting_feature, len(left_split), len(right_split)))
    
    
    #Create a leaf node if the split is perfect
    if len(left_split) == len(data):
        print('Creating a leaf node')
        return create_leaf(left_split[target])
    if len(right_split) == len(data):
        print('Creating a leaf node')
        return create_leaf(right_split[target])
    '''
    
    #Repeat on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth, min_node_size, min_error_reduction)
    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth, min_node_size, min_error_reduction)
    
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

def count_leaves(tree):
    if tree['is_leaf']:
        return 1
    return count_leaves(tree['left']) + count_leaves(tree['right'])

       
loans = pd.read_csv('Data/Week4/lending-club-data.csv')
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
train_idx = pd.read_json('Data/Week4/train-idx.json').values[:,0] #indices of the training data
valid_idx = pd.read_json('Data/Week4/validation-idx.json').values[:,0] #indices of the test data

#Turn the loans data into categorical data
loans_categorical = turn_into_categorical_variables(loans, features)

#construct training and validation data
train_data = loans_categorical.iloc[train_idx]
validation_data = loans_categorical.iloc[valid_idx]

categorical_features = loans_categorical.columns.values
categorical_features = categorical_features[categorical_features != target]

#Build a decision tree
my_decision_tree_new = decision_tree_create(train_data, categorical_features, target, max_depth = 6, 
                                min_node_size = 100, min_error_reduction=0.0)

my_decision_tree_old = decision_tree_create(train_data, categorical_features, target, max_depth = 6,
                                            min_node_size = 0, min_error_reduction=-1)

#obsere what is predicted for the first data point in the validation set
print(validation_data.iloc[0])
print ('Predicted class: %s ' % classify(my_decision_tree_new, validation_data.iloc[0]))

#print the path that was taken by the my_decision_tree_new
print('=================================================')
print('Printing the path taken by the newly learned tree')
classify(my_decision_tree_new, validation_data.iloc[0], annotate = True)

#print the path that was taken by the my_decision_tree_old
print('=================================================')
print('Printing thepath taken by the old learned tree')
classify(my_decision_tree_old, validation_data.iloc[0], annotate = True)

#Compare the classification errors on the new decision tree and old_decision_tree
error_new_tree = evaluate_classification_error(my_decision_tree_new, validation_data, target)
error_old_tree = evaluate_classification_error(my_decision_tree_old, validation_data, target)
print('classification error of the new decision tree is %s and old tree is %s' %(error_new_tree, error_old_tree))

#Exploring the effect of max_depth
model_1 = decision_tree_create(train_data, categorical_features, target, current_depth=0, max_depth=2, min_node_size=0, min_error_reduction=-1)
model_2 = decision_tree_create(train_data, categorical_features, target, current_depth=0, max_depth=6, min_node_size=0, min_error_reduction=-1)
model_3 = decision_tree_create(train_data, categorical_features, target, current_depth=0, max_depth=14, min_node_size=0, min_error_reduction=-1)

#Evaluate the above models on the train and validation data
print ("Training data, classification error (model 1):", evaluate_classification_error(model_1, train_data, target))
print ("Training data, classification error (model 2):", evaluate_classification_error(model_2, train_data, target))
print ("Training data, classification error (model 3):", evaluate_classification_error(model_3, train_data, target))
print ("Training data, classification error (model 1):", evaluate_classification_error(model_1, validation_data, target))
print ("Training data, classification error (model 2):", evaluate_classification_error(model_2, validation_data, target))
print ("Training data, classification error (model 3):", evaluate_classification_error(model_3, validation_data, target))

#Evaluate model complexity for each of the above models
#model complexity is the number of leaf nodes in the tree model
print("Number of leaves in model_1: ", count_leaves(model_1))
print("Number of leaves in model_2: ", count_leaves(model_2))
print("Number of leaves in model_3: ", count_leaves(model_3))


#Explore the effect of min_error
model_4 = decision_tree_create(train_data, categorical_features, target, current_depth=0, max_depth=6, min_node_size=0, min_error_reduction=-1)
model_5 = decision_tree_create(train_data, categorical_features, target, current_depth=0, max_depth=6, min_node_size=0, min_error_reduction=0)
model_6 = decision_tree_create(train_data, categorical_features, target, current_depth=0, max_depth=6, min_node_size=0, min_error_reduction=5)

print ("Validation data, classification error (model 4):", evaluate_classification_error(model_4, validation_data, target))
print ("Validation data, classification error (model 5):", evaluate_classification_error(model_5, validation_data, target))
print ("Validation data, classification error (model 6):", evaluate_classification_error(model_6, validation_data, target))

#Evaluate model complexity for each of the above models
#model complexity is the number of leaf nodes in the tree model
print("Number of leaves in model_4: ", count_leaves(model_4))
print("Number of leaves in model_5: ", count_leaves(model_5))
print("Number of leaves in model_6: ", count_leaves(model_6))


#Exploring the effect of min_node_size
model_7 = decision_tree_create(train_data, categorical_features, target, current_depth=0, max_depth=6, min_node_size=0, min_error_reduction=-1)
model_8 = decision_tree_create(train_data, categorical_features, target, current_depth=0, max_depth=6, min_node_size=2000, min_error_reduction=-1)
model_9 = decision_tree_create(train_data, categorical_features, target, current_depth=0, max_depth=6, min_node_size=50000, min_error_reduction=-1)

print ("Validation data, classification error (model 7):", evaluate_classification_error(model_7, validation_data, target))
print ("Validation data, classification error (model 8):", evaluate_classification_error(model_8, validation_data, target))
print ("Validation data, classification error (model 9):", evaluate_classification_error(model_9, validation_data, target))
print("Number of leaves in model_7: ", count_leaves(model_7))
print("Number of leaves in model_8: ", count_leaves(model_8))
print("Number of leaves in model_9: ", count_leaves(model_9))