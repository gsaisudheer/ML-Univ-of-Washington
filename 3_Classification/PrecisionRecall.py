'''
Created on 23-Apr-2017
Programming Assignment 1 of Week 6 from the Classification course of
Coursera Machine Learning Specialization
@author: sudheer
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def remove_punctuation(text):
    import string
    translator = str.maketrans({key: None for key in string.punctuation})
    return text.translate(translator)

def apply_threshold(probabilities, threshold):
    return np.where(probabilities >= threshold, +1, -1)
    #return np.apply_along_axis(lambda probability: +1 if probability > threshold else -1,axis=1,arr=probabilities)

def plot_pr_curve(precision, recall, title):
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis = 'x', nbins = 5)
    plt.plot(precision, recall, 'b-', linewidth=4.0, color = '#B0017F')
    plt.title(title)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.rcParams.update({'font.size': 16})
    plt.show()

products = pd.read_csv('Data/Week6/amazon_baby.csv')
products = products.fillna({'review':''})
products['review_clean'] = products['review'].apply(remove_punctuation)

#Remove all the reviews with rating = 3, since they tend to have neutral sentiment
products = products[products['rating'] != 3]

#Assign a sentiment of 1 if rating is greater than 3, else -1
products['sentiment'] = products['rating'].apply(lambda rating: +1 if rating > 3 else -1 )

#Load the train and test indices
train_idx = pd.read_json('Data/Week6/train-idx.json').values[:,0]
test_idx = pd.read_json('Data/Week6/test-idx.json').values[:,0]

#Construct the train and test data
train_data = products.iloc[train_idx] #Extract the data using the indices
test_data = products.iloc[test_idx] #Extract the data using the indices

#Use this token pattern to keep single-letter words
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
#First learn vocabulary from the training data and assign columns to words
#Then convert the training data into a sparse matrix
train_matrix = vectorizer.fit_transform(train_data['review_clean'])
#Convert the test data into a sparse matrix, using the same word-column mapping
test_matrix = vectorizer.transform(test_data['review_clean'])


#Test a sentiment classifier with logistic Regression
sentiment_model = LogisticRegression()
sentiment_model.fit(train_matrix, train_data['sentiment']) #Fit the model to the training data

#Compute the accuracy on the test data
accuracy = accuracy_score(y_true=test_data['sentiment'], y_pred=sentiment_model.predict(test_matrix))
print('Test accuracy: %s' %accuracy)

#A good model should beat the majority class classifier.
#Majority class in this dataset is the positve class.
baseline = len(test_data[test_data['sentiment'] == +1])/len(test_data)
print('Baseline accuracy (majority class classifier): %s' %baseline)

#Construct the confusion matrix
cmat = confusion_matrix(y_true=test_data['sentiment'].as_matrix(),
                         y_pred=sentiment_model.predict(test_matrix), 
                         labels= sentiment_model.classes_)

print('target_label | predicted_label | count')
print('-------------+-----------------+------')
#Print out the confusion matrix
for i, target_label in enumerate(sentiment_model.classes_):
    for j, predicted_label in enumerate(sentiment_model.classes_):
        print('{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i,j]))
        
precision = precision_score(y_true=test_data['sentiment'].as_matrix(),
                             y_pred=sentiment_model.predict(test_matrix) )

print('Precision on test data is %s' %precision)

recall = recall_score(y_true=test_data['sentiment'], y_pred=sentiment_model.predict(test_matrix))

print('Recall on test data is %s' %recall )

#Vary the threshold
probabilities = sentiment_model.predict_proba(test_matrix)[:,1] #for class +1
threshold_5_prediction = apply_threshold(probabilities, 0.5)
threshold_9_prediction = apply_threshold(probabilities, 0.9)

print('Precision on test data with 0.5 threshold: %s' %precision_score(y_true=test_data['sentiment'].as_matrix(),
                                                                        y_pred=threshold_5_prediction))
print('Recall on test data with 0.5 threshold: %s' %recall_score(y_true=test_data['sentiment'].as_matrix(),
                                                                        y_pred=threshold_5_prediction))
print('Precision on test data with 0.9 threshold: %s' %precision_score(y_true=test_data['sentiment'].as_matrix(),
                                                                        y_pred=threshold_9_prediction))
print('Recall on test data with 0.9 threshold: %s' %recall_score(y_true=test_data['sentiment'].as_matrix(),
                                                                        y_pred=threshold_9_prediction))

#Precision recall Curve
threshold_values = np.linspace(0.5,1,num=100)

#For threshold in [0.5,1] compute the precision and recall
precision_all = []
recall_all = []
for threshold in threshold_values:
    precision = precision_score(y_true=test_data['sentiment'].as_matrix(),
                                          y_pred=apply_threshold(probabilities,threshold))
    recall = recall_score(y_true=test_data['sentiment'].as_matrix(),
                                    y_pred=apply_threshold(probabilities, threshold))
    print('Threshold: %s, Precision: %s, Recall: %s' %(threshold,precision,recall))
    precision_all.append(precision)
    recall_all.append(recall)
 
plot_pr_curve(precision_all, recall_all, 'Precision recall curve (all)')

#Among all the threshold values, what is the smallest value that achieves a precision of 
#96.5 percent or higher