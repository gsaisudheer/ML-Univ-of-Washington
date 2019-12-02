'''
Created on 29-Nov-2016
Programming Assignment 1 of Week 1 from the
Coursera Machine Learning Specialization
@author: sudheer
'''
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn import linear_model
import numpy as np
from itertools import compress


def compute_accuracy(actual_sentiment, predicted_sentiment):
    correctlyClassified = (actual_sentiment == predicted_sentiment).sum()
    return (1.0*correctlyClassified)/len(actual_sentiment.index)


def remove_punctuation(text):
    import string
    translator = str.maketrans({key: None for key in string.punctuation})
    return text.translate(translator)

products = pd.read_csv('Data/Week1/amazon_baby.csv')
products = products.fillna({'review':''}) #Fill in NAs in the review column
products['review_clean'] = products['review'].apply(remove_punctuation)

#Remove all the reviews with rating=3, since they tend to have neutral sentiment
products = products[products['rating'] !=3 ]
products['sentiment'] = products['rating'].apply(lambda rating: +1 if rating > 3 else -1)

train_idx = pd.read_json('Data/Week1/module-2-assignment-train-idx.json').values[:,0] #indices of the training data
test_idx = pd.read_json('Data/Week1/module-2-assignment-test-idx.json').values[:,0] #indices of the test data

train_data = products.iloc[train_idx] #Extract the train data using the indices
test_data = products.iloc[test_idx] #Extract the test data using the indices

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b') # The pattern \w is used to match a single alphanumeric character
train_matrix = vectorizer.fit_transform(train_data['review_clean']) # Constructs a sparse matrix where (row,col) is the word count
test_matrix = vectorizer.transform(test_data['review_clean'])

sentiment_model = linear_model.LogisticRegression() # Construct a simple logistic classifier
sentiment_model.fit(train_matrix, train_data['sentiment'])

#Quiz Q1:
coefficients = sentiment_model.coef_[0]
#print('The number of non-zero coefficients are ' + str(sum(x >=0 for x in coefficients)))
#print('The number of non-zero coefficients are ' + str(len(sentiment_model.coef_[0] >= 0)))
print(sum(coefficients >= 0))



'''Making predictions with logistic regression'''
sample_test_data = test_data.iloc[10:13]
#Compute the scores for the sample_test_data
#First obtain the sparse matrix representation for the review_clean column of the data
sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
scores = sentiment_model.decision_function(sample_test_matrix)
print(scores)

'''Predicting Sentiment'''
#compute the predicted labels for sample_test_data
sample_test_data.loc[:,'scores'] = scores
sample_test_data['predicted_sentiment'] = sample_test_data['scores'].apply(lambda score: +1 if score > 0 else -1)

'''Probability predictions
P(y = +1|xi, w) = 1/(1+exp(-wT xi))
'''
sample_test_data['probability'] = sample_test_data['scores'].apply(lambda score: 1.0/(1+np.exp(-score)))
#Quiz Q2:
#sample_test_data = sample_test_data.sort(columns='probability',ascending = False)
print('Of the sample test data, the one with lowest prob of being classified as positive is ')
print(sample_test_data.loc[:,('name','probability')])


'''Find the most positive and most negative reviews
Examine the full test data '''
test_data.loc[:,'predicted_sentiment'] = sentiment_model.predict(test_matrix)
test_data.loc[:,'probabilities'] = sentiment_model.predict_proba(test_matrix)[:,1] # [1] gives the probabilities that the output is +1.
test_data_sorted = test_data.sort(columns = 'probabilities', ascending=False)
#Quiz Q3:
print('Names of the products having most positive reviews: ')
print(test_data_sorted.head(20).loc[:,('name','probabilities')])

#Quiz Q4:
print('Names of the products having most negative reviews: ')
print(test_data_sorted.tail(20).loc[:,('name','probabilities')])

'''Evaluate the accuracy of the trained classifier'''
#Quiz Q5:
print('Accuracy of the sentiment_model on the test data is ' + str(compute_accuracy(test_data.sentiment, test_data.predicted_sentiment)))
#Quiz Q6:
#A higher accuracy value on the training_data does not imply that the classifier is always better

'''Learn Another classifier with fewer words'''
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']
vectorizer_word_subset =  CountVectorizer(vocabulary=significant_words)
train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_data['review_clean'])
test_matrix_word_subset = vectorizer_word_subset.transform(test_data['review_clean'])
#Construct a simple logistic classifier with the above vocabulary
simple_model = linear_model.LogisticRegression()
simple_model.fit(train_matrix_word_subset,train_data['sentiment'])
simple_model_coef_table = pd.DataFrame({'word':significant_words,
                                        'coefficient':simple_model.coef_.flatten()})
simple_model_coef_table = simple_model_coef_table.sort(columns='coefficient',ascending=False)

#Quiz Q7
simple_model_positive_words = simple_model_coef_table[simple_model_coef_table['coefficient'] >= 0]['word']
print('Printing the words which have positive coefficients ')
print(simple_model_positive_words)

sentiment_model_vocab = list(vectorizer.vocabulary_.keys())
positive_values = (sentiment_model.coef_ >= 0)[0]
sentiment_model_positive_words = compress(sentiment_model_vocab,positive_values)
#Quiz Q8
containsAll = set(simple_model_positive_words).issubset(set(sentiment_model_positive_words))
if containsAll:
    print('All positive words from simple_model ARE POSITIVE in sentiment_model ')
else:
    print('All positive words from simple model ARE NOT POSITIVE in sentiment_model ')

'''
sentiment_model_vocab = np.array(list(vectorizer.vocabulary_.keys()))
positive_values = np.array((sentiment_model.coef_ > 0)[0])
print(positive_values[0:10])
print(np.shape(positive_values))
print(np.shape(sentiment_model_vocab))
sentiment_model_positive_words = sentiment_model_vocab[positive_values]
print(set(simple_model_positive_words).issubset(set(sentiment_model_positive_words)))
'''

'''Compare the accuracy of the sentiment_model vs the simple_model'''
#Compute the classification accuracy of sentiment_model on the train data
accuracy_sentiment_model_train_data = compute_accuracy(train_data['sentiment'], sentiment_model.predict(train_matrix))
print('Accuracy of the sentiment_model on the train data is ' + str(accuracy_sentiment_model_train_data))
#Compute the classification accuracy of simple_model on the train data
accuracy_simple_model_train_data = compute_accuracy(train_data['sentiment'], simple_model.predict(train_matrix_word_subset))
print('Accuracy of the simple_model on the train data is ' + str(accuracy_simple_model_train_data))
#Compute the classification accuracy of the sentiment_model on the test data
accuracy_sentiment_model_test_data = compute_accuracy(test_data['sentiment'], sentiment_model.predict(test_matrix))
print('Accuracy of the sentiment_model on the test data is ' + str(accuracy_sentiment_model_test_data))
#Compute the classification accuracy of the simple_model on the test data
accuracy_simple_model_test_data = compute_accuracy(test_data['sentiment'], simple_model.predict(test_matrix_word_subset))
print('Accuracy of the simple_model on the test data is ' + str(accuracy_simple_model_test_data))


'''Compare with the baseline prediction, i.e majority class prediction'''
num_positive_sentiments_train = len(train_data.loc[train_data.sentiment == 1].index)
num_negative_sentiments_train = len(train_data.index) - num_positive_sentiments_train
print('num positive sentiments train ' + str(num_positive_sentiments_train))
print('num negative sentiments train ' + str(num_negative_sentiments_train))
if num_positive_sentiments_train > num_negative_sentiments_train:
    majority_prediction = 1
else:
    majority_prediction = -1
accuracy_majority_model_test_data = compute_accuracy(test_data['sentiment'], majority_prediction)
print('Accuracy of the majority_model on the test data is ' + str(accuracy_majority_model_test_data))
    













