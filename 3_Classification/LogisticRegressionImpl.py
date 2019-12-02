'''
Created on 24-Dec-2016
Programming Assignment 1 of Week 2 from the 
Coursera Machine Learning Specialization
@author: sudheer
'''

import pandas as pd
import numpy as np

def remove_punctuation(text):
    import string
    translator = str.maketrans({key: None for key in string.punctuation})
    return text.translate(translator)

#Take the dataframe and return the features_matrix and the outputLabel_matrix
def get_numpy_data(dataframe, features, label):
    dataframe['constant'] = 1
    features = np.append(['constant'], features)
    features_frame = dataframe[features]
    features_matrix = features_frame.as_matrix()
    label_sarray = dataframe[label]
    label_array = label_sarray.as_matrix()  
    return (features_matrix, label_array)

def predict_probability(feature_matrix, coefficients):
    #feature_matrix is NxD and coefficients is a D-dim vector
    score = np.dot(feature_matrix,coefficients)
    #compute the prob using the link/sigmoid function
    probabilities = 1./(1+np.exp(-score))
    return probabilities

def feature_derivative(errors,feature):
    #Compute the dot product between errors and feature
    derivative = np.dot(errors,feature)
    return derivative

def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment == +1)
    scores = np.dot(feature_matrix,coefficients)
    lp = np.sum((indicator - 1)*scores - np.log(1. + np.exp(-scores)))
    return lp

def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    coefficients = np.array(initial_coefficients)
    for itr in range(max_iter):
        predictions = predict_probability(feature_matrix, coefficients)
        indicator = (sentiment == +1)
        #Compute errors as indicator - predictions
        errors = indicator - predictions
        for j in range(len(coefficients)):
            #Compute the feature derivative
            derivative = feature_derivative(errors, feature_matrix[:,j])
            #Add the step size times the derivative to the current coefficient
            coefficients[j] = coefficients[j] + step_size * derivative
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print ('iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp))
    return coefficients
    
products = pd.read_csv('Data/Week2/amazon_baby_subset.csv')
important_words = pd.read_json('Data/Week2/important_words.json').values[:,0]
products = products.fillna({'review':''})  # fill in N/A's in the review column

#Add a column review_clean to the products DataFrame by removing the Punctuation
products['review_clean'] = products['review'].apply(remove_punctuation)
for word in important_words:
    products[word] = products['review_clean'].apply(lambda s: s.split().count(word))

products['contains_perfect'] = products['perfect'].apply(lambda s: +1 if s > 0 else 0)
#Quiz Q1
print('Number of reviews containing the word perfect are ' + str(products['contains_perfect'].sum()))

feature_matrix, sentiment = get_numpy_data(products, important_words, 'sentiment')
#Quiz Q2
print('Number of features in the feature_matrix are ' + str(np.size(feature_matrix,1)))

coefficients = logistic_regression(feature_matrix, sentiment, np.zeros(194), 1e-7 , 301)

#Perform predictions using coefficients that were computed above
scores = np.dot(feature_matrix,coefficients)
#Quiz Q5
print('Number of reviews predicted to have positive_sentiment ' + str(np.sum(scores > 0)))

#Quiz Q6
predicted_sentiment = np.where(scores > 0, +1, -1)
correctly_classified = np.sum(predicted_sentiment == sentiment)
accuracy = correctly_classified/np.size(predicted_sentiment)
print('Accuracy of the model is ' + str(accuracy))

coefficients = list(coefficients[1:]) #exclude the intercept term
word_coefficient_tuples = [(word,coefficient) for word,coefficient in zip(important_words,coefficients)]
word_coefficient_tuples = sorted(word_coefficient_tuples, key=lambda x: x[1], reverse=True)

#Quiz Q7
print('The 10 most positive words are ')
print(word_coefficient_tuples[0:10])

#Quiz Q8
print('The 10 most negative words are ')
print(word_coefficient_tuples[-10:])


