'''
Created on 28-Dec-2016
Programming Assignment 2 of Week 2 from the Classification course of
Coursera Machine Learning Specialization
@author: sudheer
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 10, 6

def remove_punctuation(text):
    import string
    translator = str.maketrans({key: None for key in string.punctuation})
    return text.translate(translator)

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

def feature_derivative_with_L2(errors, feature, coefficient, l2_penalty, feature_is_constant):
    derivative = np.dot(errors,feature)
    if not feature_is_constant:
        derivative = derivative - 2 * l2_penalty * coefficient
    return derivative

def compute_log_likelihood_with_L2(feature_matrix, sentiment, coefficients, l2_penalty):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    lp = np.sum((indicator-1)*scores - np.log(1. + np.exp(-scores))) - l2_penalty*np.sum(coefficients[1:]**2)
    return lp

def logistic_regression_with_L2(feature_matrix, sentiment, initial_coefficients, step_size, l2_penalty, max_iter):
    coefficients = np.copy(initial_coefficients) #use np.copy otherwise it'll be treated like a reference
    for itr in range(max_iter):
        predictions = predict_probability(feature_matrix, coefficients)
        indicator = (sentiment == +1)
        errors = indicator - predictions
        for j in range(len(coefficients)):
            is_intercept = (j==0)
            derivative = feature_derivative_with_L2(errors, feature_matrix[:,j], coefficients[j], l2_penalty, is_intercept)
            coefficients[j] = coefficients[j] + step_size * derivative
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood_with_L2(feature_matrix, sentiment, coefficients, l2_penalty)
            print ('iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp))
    return coefficients
    
    
def make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list):
    cmap_positive = plt.get_cmap('Reds')
    cmap_negative = plt.get_cmap('Blues')
    xx = l2_penalty_list
    plt.plot(xx, [0.]*len(xx), '--',lw=1,color='k')
    table_positive_words = table[table['features'].isin(positive_words)]
    table_negative_words = table[table['features'].isin(negative_words)]
    del table_positive_words['features']
    del table_negative_words['features']
    for i in range(len(positive_words)):
        color = cmap_positive(0.8*((i+1)/(len(positive_words)*1.2)+0.15))
        plt.plot(xx, table_positive_words[i:i+1].as_matrix().flatten(),
                 '-',label = positive_words[i],linewidth = 4.0, color=color)
    for i in range(len(negative_words)):
        color = cmap_negative(0.8*((i+1)/(len(negative_words)*1.2)+0.15))
        plt.plot(xx, table_negative_words[i:i+1].as_matrix().flatten(),
                 '-',label = negative_words[i],linewidth = 4.0, color = color)
    
    plt.legend(loc='best', ncol=3, prop={'size':16}, columnspacing=0.5)
    plt.axis([1, 1e5, -1, 2])
    plt.title('Coefficient path')
    plt.xlabel('L2 penalty ($\lambda$)')
    plt.ylabel('Coefficient value')
    plt.xscale('log')
    plt.rcParams.update({'font.size': 18})
    plt.tight_layout()
    plt.show()

def make_predictions(feature_matrix,coefficients):
    scores = np.dot(feature_matrix,coefficients)
    predicted_sentiment = np.where(scores > 0, +1, -1)
    return predicted_sentiment

def get_classification_error(feature_matrix,coefficients,output):
    predicted_output = make_predictions(feature_matrix, coefficients)
    misclassified = np.sum(predicted_output != output)
    numExamples = len(output)
    return (misclassified*1.0)/len(output)
    
    
products = pd.read_csv('Data/Week2/2b/amazon_baby_subset.csv')
important_words = pd.read_json('Data/Week2/2b/important_words.json').values[:,0]

products = products.fillna({'review':''}) #Fill in NAs in the review column

#Make a new column review_clean which contains the reviews with the punctuation removed
products['review_clean'] = products['review'].apply(remove_punctuation)
for word in important_words:
    products[word] = products['review_clean'].apply(lambda s: s.split().count(word))
    
#Read the indices of training and test data
train_idx = pd.read_json('Data/Week2/2b/module-4-assignment-train-idx.json').values[:,0]
validation_idx = pd.read_json('Data/Week2/2b/module-4-assignment-validation-idx.json').values[:,0]

train_data = products.iloc[train_idx]
validation_data = products.iloc[validation_idx]

feature_matrix_train, sentiment_train = get_numpy_data(train_data,important_words,'sentiment')
feature_matrix_valid, sentiment_valid = get_numpy_data(validation_data, important_words, 'sentiment')

l2_penalties = [0,4,10,1e2,1e3,1e5]
feature_matrix = feature_matrix_train
sentiment = sentiment_train
initial_coefficients = np.zeros(194)
step_size = 5e-6
max_iter = 501

l2_penalty_names = []
l2_penalty_names.append('features')
for penalty in l2_penalties:
    l2_penalty_names.append('coefficients_'+str(penalty)+'_penalty')

#coeffs_array = np.empty(np.shape(l2_penalties)[0])
l2_penalties_coeffs = pd.DataFrame(columns = l2_penalty_names)
features_with_constant = np.append(np.array(['constant']),important_words)
l2_penalties_coeffs['features'] = features_with_constant

for penalty in l2_penalties:
    l2_penalties_coeffs['coefficients_'+str(penalty)+'_penalty'] = logistic_regression_with_L2(feature_matrix, sentiment, initial_coefficients, step_size, penalty, max_iter)

#Quiz Q2:
#Using 0 penalty, find the most positive and most negative words
l2_penalties_0_coeff = (l2_penalties_coeffs.loc[:,['features','coefficients_0_penalty']]).sort_values('coefficients_0_penalty',ascending=True)
#l2_penalties_0_coeff =  (l2_penalties_0_coeff.iloc[0:5]).sort_values('coefficients_0_penalty',ascending=True)
#l2_penalties_0_coeff = l2_penalties_0_coeff.sort_values('coefficients_0_penalty',ascending=True)
positive_words = l2_penalties_0_coeff['features'].iloc[-5:].as_matrix().flatten()
negative_words = l2_penalties_0_coeff['features'].iloc[0:5].as_matrix().flatten()
print('Most negative words are ')
print(negative_words)
print('Most positive words are ')
print(positive_words)

make_coefficient_plot(l2_penalties_coeffs, positive_words, negative_words, l2_penalties)
#l2_penalties_0_coeff = l2_penalties_coeffs.loc[:,['features','coefficients_0_penalty']].sort_values('coefficients_0_penalty',ascending=True))

#Measuring Accuracy of the classifier on the training and validation data
train_accuracy = pd.DataFrame(columns = ['l2_penalty','classification_error'])
train_accuracy['l2_penalty'] = l2_penalties
errors = []
for penalty in l2_penalties:
    errors.append(get_classification_error(feature_matrix_train, l2_penalties_coeffs['coefficients_'+str(penalty)+'_penalty'], sentiment_train))
train_accuracy['classification_error'] = errors

valid_accuracy = pd.DataFrame(columns = ['l2_penalty','classification_error'])
valid_accuracy['l2_penalty'] = l2_penalties
errors = []
for penalty in l2_penalties:
    errors.append(get_classification_error(feature_matrix_valid, l2_penalties_coeffs['coefficients_'+str(penalty)+'_penalty'], sentiment_valid))
valid_accuracy['classification_error'] = errors


print('Printing Training errors')
print(train_accuracy)
print('Printing Validation errors')
print(valid_accuracy)