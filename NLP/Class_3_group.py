
# %% Imports and Helper functions

import re
import collections
import stanza
import requests
import random

# %% Download and read txt
## recommmended site for urls: http://www.glozman.com/textpages.html
def download_txt(url, name=None, write=True):
    if write and name is None:
        raise ValueError("Name is None")
    r = requests.get(url)

    # write content to a .txt file
    with open(name, 'w') as f:
        f.write(r.content.decode("windows-1252"))

url = "http://www.glozman.com/TextPages/Frank%20Herbert%20-%20Dune.txt"
name = "Dune.txt"

download_txt(url, name)
txt = ""

Dune = []
with open(name, "r") as File1:
    for line in File1:
        line.strip()
        if line == '\n':
            continue
        Dune.append(line)

Dune = " ".join(Dune)
Dune = Dune.replace("\n","")
Dune = Dune.replace("\'", "'")

#%% ((?<=[!.?:\(\)])\s+)|(\s){2,}
def sentence_segment(txt):
    """
    txt (str): Text which you want to be segmented into sentences.

    Example:
    >>> txt = "NLP is very cool. It is also useful"
    >>> sentence_segment(txt)
    ["NLP is very cool", "It is also useful"]
    """
    stringing = re.split("((?<=[!.?:\(\)])\s+)|(\s){2,}", txt)
    return [f"<s> {string} <\s>" for string in stringing]


def tokenize(sentences):
    """
    sentences (list): Sentences which you want to be tokenized

    Example:
    >>> sent = ["NLP is very cool", "It is also useful"]
    >>> tokenize(sent)
    [["NLP", "is", "very", "cool"], ["It", "is", "also", "useful"]]
    """
    split_list = [re.split(" ", word) for word in sentences]
    return [item for sublist in split_list for item in sublist]

#the list: ["<s> the first sentence <\S>", "<s> the second one <\S>"]

def n_grams(tokenlist, n):
    """
    tokenlist (list): A list of tokens
    n (int): Indicate the n in n-gram. n=2 denotes bigrams

    creates n-grams from a given tokenlist

    Example:
    >>> tokens = ["NLP", "is", "very", "cool"]
    >>> n_grams(tokens, n=2)
    [["NLP", "is"], ["is", "very"], ["very", "cool"]]
    """
    if n == 1:
        return tokenlist
    empty_list = []
    
    for i in range(len(tokenlist)-1):
        new_list = tuple(tokenlist[i:i+n])

        if len(new_list) == n:
            empty_list.append(new_list)
            
    return empty_list

# %% Write a program to compute unsmoothed unigrams and bigrams:

Our_grams_bi = n_grams(tokenize(sentence_segment(Dune)),2)
Our_grams_uni = n_grams(tokenize(sentence_segment(Dune)),1)

# %%
def frequency(n_gram):
    return dict(collections.Counter(n_gram))

our_bigram = frequency(Our_grams_bi)
our_unigram = frequency(Our_grams_uni)

# %% 
def prob(bi_dictionary, uni_dictionary):
    return {key:bi_dictionary.get(key)/uni_dictionary.get(key[0]) for key in bi_dictionary}
# %%
computes = prob(our_bigram, our_unigram)

# %%
sorted_uni = sorted(our_unigram, key = our_unigram.get, reverse = True)

# %%
sorted_bi = sorted(our_bigram, key = our_bigram.get, reverse = True)

# %% [(this, that)]
def random_generator(dictionary, start_word = "<s>", n =2):
    sentence = " ".join(start_word)
    
    while start_word[-1] != "<\\s>":
        candidate_dict = {word:dictionary.get(word) for word in dictionary if word[:-1] == start_word}
        next_word = random.choices(list(candidate_dict.keys()), weights = list(candidate_dict.values()), k=1)[0]
        sentence = f"{sentence} {next_word[1]}"
        start_word = next_word[1:]
    return sentence


#random_generator(n_prob(3), ("The", "man"), 3)

# %% for n-grams

def n_prob(n):
    n_dict = frequency(n_grams(tokenize(sentence_segment(Dune)),n))
    sub_dict = frequency(n_grams(tokenize(sentence_segment(Dune)),n-1))
    return {key:n_dict.get(key)/sub_dict.get(key[:-1]) for key in n_dict}


# %%
random_generator(n_prob(4), ("The", "man", "said"), n = 4)
# %%
