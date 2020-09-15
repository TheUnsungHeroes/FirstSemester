import collections
import re
import requests


def download_txt(url, name=None, write=True):
    if write and name is None:
        raise ValueError("Name is None")
    r = requests.get(url)

    # write content to a .txt file
    with open(name, 'w') as f:
        f.write(r.content.decode("windows-1252"))


def prep_text(text):
    sep_text = re.split(r"(?:[;\?\.:!_\v]+[\s])", text)
    return [f"<s> {string} </s>" for string in sep_text]


def tokenize(text):
    split_list = [re.split(" ", string) for string in text]
    return [item for sublist in split_list for item in sublist]


def n_grams(tokenlist, n):
    empty_list = []

    if n == 1:
        return tokenlist

    for i in range(len(tokenlist)-1):
        new_list = tuple(tokenlist[i:i+n])

        if len(new_list) == n:
           empty_list.append(new_list)
    
    return empty_list


def count_n_grams(ngram):
    return dict(collections.Counter(ngram))


def count_to_probability(bi_dict, uni_dict):
    return {x : bi_dict[x]/uni_dict.get(x[0]) for x in bi_dict}


#PIPELINE:
url = 'http://www.glozman.com/TextPages/03%20-%20The%20Return%20Of%20The%20King.txt'  # insert url of book you want to scrape
book_name = "lotr3.txt"  # call it something
#download_txt(url, bookname)  # download it

with open(book_name, "r") as f:
        txt = f.read()[360:]  # change according to need
        txt = txt.rstrip()  # strip whitespace

prepped_txt = prep_text(txt)
prepped_txt = tokenize(prepped_txt)

#UNI:
prepped_txt_uni = n_grams(prepped_txt, 1)
prepped_txt_uni = count_n_grams(prepped_txt_uni)

#BI:
prepped_txt = n_grams(prepped_txt, 2)
prepped_txt = count_n_grams(prepped_txt)

#PROBABILITIES

prepped_prob = count_to_probability(prepped_txt, prepped_txt_uni)

sorted_prepped_prob = sorted(prepped_prob, key=prepped_prob.get, reverse=True)

sorted_freqs = sorted(prepped_txt, key=prepped_txt.get, reverse=True)

# NOTE:
''' The first 4 most frequent cases are completely bogus. Instead,
it makes more sense to look at non-trivial cases. Almost the same can be said
for the probabilities.
The skew is due to some combinations only occuring once.'''

sorted_freqs[5:15]
sorted_prepped_prob[30:40]