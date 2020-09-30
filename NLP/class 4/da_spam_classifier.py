#%%

import pandas as pd
import numpy as np
from sklearn import model_selection as model_selection
from text_processor import Text
from text_processor_functions import *


#%% read data and rename columns
data = pd.read_csv("spam.csv", encoding="cp1252", usecols = ["v1", "v2"])
data.columns = ["category", "sentence"]
#data['class'].value_counts(normalize=True)
#let's split the data into training and testing sets
#rain, test = model_selection.train_test_split(data)
#let's look at the data a little bit: how many spams, how many hams
#len([item for item in data["v1"] if item == "ham"])

# %%


## PREPROCESSING:

def NaiveBayes(dataframe):

    ## PRIORS:
    spam_count = (dataframe.category == "spam").sum()
    ham_count = (dataframe.category == "ham").sum()
    total_count = len(dataframe)

    spam_prior = spam_count / total_count
    ham_prior = ham_count / total_count

    ## INITIALIZE DICTIONARIES:
    spam_words = {}
    ham_words = {}
    

    train, test = model_selection.train_test_split(data)
    
    spam = train[train["category"]== "spam"]
    ham = train[train["category"]== "ham"]

    all_words = Text(' '.join(dataframe['sentence'].tolist()))
    word_freq = all_words.get_freq()


    return word_freq
    
    #for i, row in dataframe.iterrows():
    #    new_sen, current_category = Text(row["sentence"]), row["category"]
    #    for sentence in new_sen.tokens:
    #        word_freq = token_frequencies(sentence)
            
    
        
    #print(spam_count)
    #print(ham_count)
    #print(total_count)

NaiveBayes(data)
#train[0:10]
#%%
#
#data[data["category"] == "spam"]
#%%

#let's get the priors for the classes (count freq of spam and ham in training set)
#len(train) #4179: number of all classes
#(train.category == "spam").sum() #552

# %% 
##let's train the model :) 









































# def naive_bayes(D):
#     #initialize 
#     spams = 0
#     hams = 0
#     spam_words = []
#     ham_words = []
#     split_text = re.split("[\s]*\n[\s]*", D.strip())
#     total_count = len(split_text)
#     all_words = get_words(split_text)
#     log_likelihood = {}

#     for i in split_text:
#         if i[0] == "+":
#             positives += 1
#             text = re.split("[\s]*\n[\s]*", i.strip())
#             positive_words.append(i)
#         else:
#             negatives += 1
#             text = re.split("[\s]*\n[\s]*", i.strip())
#             negative_words.append(i)


#     log_prior = {"+": positives/total_count, "-": negatives/total_count}
    
#     positive_words = get_words(positive_words)
#     negative_words = get_words(negative_words)

#     counts_total = token_frequencies(all_words)
#     counts_positive = token_frequencies(positive_words)
#     counts_negative = token_frequencies(negative_words)

#     cardinal = len(counts_total)
#     sum_pos = sum(counts_positive.values())
#     sum_neg = sum(counts_negative.values())

#     pos_dict = {}
#     neg_dict = {}

#     for word in all_words:
#         if counts_positive.get(word) != None:
#             pos_freq = counts_positive[word]
#         else: 
#             pos_freq = 0
#         if counts_negative.get(word) != None:
#             neg_freq = counts_negative[word]
#         else:
#             neg_freq = 0
#         pos_dict[word] = (pos_freq+1)/(sum_pos + cardinal)
#         neg_dict[word] = (neg_freq+1)/(sum_neg + cardinal)


#     return (log_prior, pos_dict, neg_dict, list(counts_total.keys()))


# def test_naive_bayes(testdoc, logprior, log_like_pos, log_like_neg, V):
#     positive,negative = logprior.values()
#     tokenized_test = tokenize(sentence_segment(testdoc))
    
#     for i in tokenized_test[0]:
#         if i in V:
#                 positive = positive * log_like_pos.get(i)
#                 negative = negative * log_like_neg.get(i)

#     if positive > negative:
#         return ("positive", positive)
#     else:
#         return ("negative", negative)

# log_prior, log_like_pos, log_like_neg, V = naive_bayes(training_set)

# test_naive_bayes(test_set, log_prior, log_like_pos, log_like_neg, V)