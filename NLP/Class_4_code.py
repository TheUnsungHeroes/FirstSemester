
import re
import math
from text_processor_functions_mikkel import *

training_set = '''
- just totally dull 
- completely predictable and lacking energy
- no surprises and very few laughs 
+ very profound 
+ the most fun film of the summer 
'''

test_set = "predictable with no originality" 

### JUST TO GET A HANG OF IT: CALCULATION FOR "the" maximum likelihood:

#P("the"| +) = 2+1/(9+20) = 3/29 (!!!) which is what we get.

def get_words(sentences):
    return [item for sublist in tokenize(sentences) for item in sublist if item != ""]

def naive_bayes(D):
    #initialize 
    positives = 0
    negatives = 0
    positive_words = []
    negative_words = []
    split_text = re.split("[\s]*\n[\s]*", D.strip())
    total_count = len(split_text)
    all_words = get_words(split_text)
    log_likelihood = {}

    for i in split_text:
        if i[0] == "+":
            positives += 1
            text = re.split("[\s]*\n[\s]*", i.strip())
            positive_words.append(i)
        else:
            negatives += 1
            text = re.split("[\s]*\n[\s]*", i.strip())
            negative_words.append(i)


    log_prior = {"+": positives/total_count, "-": negatives/total_count}
    
    positive_words = get_words(positive_words)
    negative_words = get_words(negative_words)

    counts_total = token_frequencies(all_words)
    counts_positive = token_frequencies(positive_words)
    counts_negative = token_frequencies(negative_words)

    cardinal = len(counts_total)
    sum_pos = sum(counts_positive.values())
    sum_neg = sum(counts_negative.values())

    pos_dict = {}
    neg_dict = {}

    for word in all_words:
        if counts_positive.get(word) != None:
            pos_freq = counts_positive[word]
        else: 
            pos_freq = 0
        if counts_negative.get(word) != None:
            neg_freq = counts_negative[word]
        else:
            neg_freq = 0
        pos_dict[word] = (pos_freq+1)/(sum_pos + cardinal)
        neg_dict[word] = (neg_freq+1)/(sum_neg + cardinal)


    return (log_prior, pos_dict, neg_dict, list(counts_total.keys()))


def test_naive_bayes(testdoc, logprior, log_like_pos, log_like_neg, V):
    positive,negative = logprior.values()
    tokenized_test = tokenize(sentence_segment(testdoc))
    
    for i in tokenized_test[0]:
        if i in V:
                positive = positive * log_like_pos.get(i)
                negative = negative * log_like_neg.get(i)

    if positive > negative:
        return ("positive", positive)
    else:
        return ("negative", negative)

log_prior, log_like_pos, log_like_neg, V = naive_bayes(training_set)

test_naive_bayes(test_set, log_prior, log_like_pos, log_like_neg, V)