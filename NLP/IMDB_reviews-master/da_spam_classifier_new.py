#%%
import pandas as pd
import numpy as np
from sklearn import model_selection as model_selection
from text_processor import Text
from text_processor_functions import *
import math


#%% read data and rename columns
data = pd.read_csv("spam.csv", encoding="cp1252", usecols = ["v1", "v2"])
data.columns = ["category", "sentence"]
#data['class'].value_counts(normalize=True)
#let's split the data into training and testing sets
#rain, test = model_selection.train_test_split(data)
#let's look at the data a little bit: how many spams, how many hams
#len([item for item in data["v1"] if item == "ham"])

## SPLIT THE DATA:

train, test = model_selection.train_test_split(data)

# %%


## PREPROCESSING:

def NaiveBayes(dataframe):

    ## PRIORS:
    spam_count = (dataframe.category == "spam").sum()
    ham_count = (dataframe.category == "ham").sum()
    total_count = len(dataframe)

    spam_prior = math.log(spam_count / total_count)
    ham_prior = math.log(ham_count / total_count)

    ## INITIALIZE DICTIONARIES:
    
    spam = data[data["category"]== "spam"]
    ham = data[data["category"]== "ham"]

    ## MAKE CLASSES:

    all_words = Text(' '.join(dataframe['sentence'].tolist()))
    spam_words = Text(' '.join(spam['sentence'].tolist()))
    ham_words = Text(' '.join(ham['sentence'].tolist()))

    ## GET FREQS:

    all_word_freq = all_words.get_freq()
    spam_word_freq = spam_words.get_freq()
    ham_word_freq = ham_words.get_freq()

    ## INITIALIZE DICTIONARIES:
    spam_likelihood = {}
    ham_likelihood = {}

    ## SUMS:
    sum_spam = sum(spam_word_freq.values())
    sum_ham = sum(ham_word_freq.values())
    cardinal = len(all_word_freq)
    
    #loop for loops
    for i in list(all_word_freq.keys()):
        if spam_word_freq.get(i) != None:
            spam_word_freq[i] = spam_word_freq.get(i)+1
        else:
            spam_word_freq[i] = 1
            
        if ham_word_freq.get(i) != None:
            ham_word_freq[i] = ham_word_freq.get(i)+1
        else:
            ham_word_freq[i] = 1
        spam_likelihood[i] = math.log(spam_word_freq[i]/(sum_spam + cardinal))
        ham_likelihood[i] = math.log(ham_word_freq[i]/(sum_ham + cardinal))

    priors = {"ham": ham_prior , "spam": spam_prior}

    return (priors, spam_likelihood, ham_likelihood, list(all_word_freq.keys()))
    
#%%

trained = NaiveBayes(train)

#%%

#clean test data and run test


def test_bayes(test, priors, spam_likelihood, ham_likelihood, all_words):
    
    ## TOKENIZE:

    test_class = Text(test)
    test_tokens = [word for sublist in test_class.tokens for word in sublist]
    
    spam_prior, ham_prior = priors["spam"], priors["ham"]

    for category in priors:
        for word in test_tokens:
            if category == "spam":
                if word in all_words:
                    spam_prior += spam_likelihood[word]
            else:
                if word in all_words:
                    ham_prior += ham_likelihood[word]

    if spam_prior > ham_prior:
        return "spam"
    else:
        return "ham"
        



# %% 

#precision = TP/(TP+FP)
#Recall = TP/(TP+FN)

def apply_test(test, trained):
    return [test_bayes(x, trained[0],trained[1], trained[2], trained[3]) for x in test["sentence"]]

result = apply_test(test, trained)

pd.DataFrame(list(zip(result, list(test["category"]))), columns = ["prediction", "category"])

def evaluate(result, test):
    df = pd.DataFrame(list(zip(result, list(test["category"]))), columns = ["prediction", "category"])

    TP = sum((df["category"] == "spam") & (df["prediction"] == "spam"))
    FP = sum((df["category"] == "ham") & (df["prediction"] == "spam"))
    TN = sum((df["category"] == "ham") & (df["prediction"] == "ham"))
    FN = sum((df["category"] == "spam") & (df["prediction"] == "ham"))

    accuracy = sum(test["category"] == result) / len(test)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    
    print("Accuracy:", accuracy, "Precision:", precision, "Recall:", recall)
    return (accuracy, precision, recall)

evaluate(result, test)
#list(map(test_bayes, test["sentence"], trained[0],trained[1], trained[2], trained[3]))

































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
# %%
