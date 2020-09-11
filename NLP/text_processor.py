"""
This script contain an example Text class

Each function contains:
An explanation as well as an example
Your job as a studygroup is to make the functions in class 2 and 3.

In class 3 we will then assemble a pipeline using the Text class at the end.


I suggest you start from the top and for each function:
    1) discuss potential solution (e.g. sentence segmentation splitting by ".")
    2) implement the simplest solution first for each function
    3) go back and improve upon the initial function

for class 2 it would be ideal if you have a simple version of the following
functions:
    sentence_segment
    tokenize 
    ner_regex

Additional stuff which you might add is:
    A function for dependency parsing using stanza
    alternatives to each of the function (e.g. using tokenization using nltk)
    Add a function for stemming
    Add plotting functionality for word frequencies
    Add plotting functionality for dependency trees
"""
import re
import collections
import stanza

# %%


def sentence_segment(txt):
    """
    txt (str): Text which you want to be segmented into sentences.

    Example:
    >>> txt = "NLP is very cool. It is also useful"
    >>> sentence_segment(txt)
    ["NLP is very cool", "It is also useful"]
    """

    return re.split("(?<=[!.?:])\s(?=[A-Z])", txt)


sentence_segment("Wait, what? How did this happen?")

# %%


def tokenize(sentences):
    """
    sentences (list): Sentences which you want to be tokenized

    Example:
    >>> sent = ["NLP is very cool", "It is also useful"]
    >>> tokenize(sent)
    [["NLP", "is", "very", "cool"], ["It", "is", "also", "useful"]]
    """

    empty_list = []

    for sentence in sentences:
        new_list = re.split("\W", sentence)
        empty_list.append(new_list)
    return empty_list


tokenize(["Here are some sentences", "they could also be called i"])

# %%


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
    empty_list = []

    for i in range(len(tokenlist)-1):
        new_list = tokenlist[i:i+n]

        if len(new_list) == n:
            empty_list.append(new_list)
    
    return empty_list


n_grams(["NLP", "is", "very", "cool"], 3)

# %%


def ner_regex(tokenlist):
    """
    tokenlist (list): A list of tokens

    performs named entity recognition using regular expressions
    Example:
    >>> sent = ["Karl Friston is very cool"]
    >>> ner_regex(sent)
    ["Karl Friston"]
    """
    
    return re.findall("([A-Z][a-z]+(?:[\s]*(?:[A-Z][a-z]+))+)", *tokenlist)


ner_regex(["Karl Friston's Mom is very cool. But what about Mom? Who is she?"])


# %%


def token_frequencies(tokenlist):
    """
    tokenlist (list): A list of tokens

    return a list of tokens and their frequencies

    Example:
    >>> tokens = [["NLP", "is", "very", "cool"],
                  ["It", "is", "also", "useful"]]
    >>> token_frequencies(sent)
    {"NLP": 1, "is": 2, "very": 1, "cool": 1, "It": 1, "also": 1, "useful": 1}
    """

    if isinstance(tokenlist[0], list):
        flat_list = [item for sublist in tokenlist for item in sublist]
    else:
        flat_list = tokenlist
    
    return {item: flat_list.count(item) for item in flat_list}


token_frequencies([["wait", "wait", "do"], ["something", "something", "do"]])

# %%


def token_frequencies(tokenlist):
    """
    tokenlist (list): A list of tokens

    return a list of tokens and their frequencies

    Example:
    >>> tokens = [["NLP", "is", "very", "cool"],
                  ["It", "is", "also", "useful"]]
    >>> token_frequencies(sent)
    {"NLP": 1, "is": 2, "very": 1, "cool": 1, "It": 1, "also": 1, "useful": 1}
    """
    if isinstance(tokenlist[0], list):
        flat_list = [item for sublist in tokenlist for item in sublist]

    else:
        flat_list = tokenlist
    
    return dict(collections.Counter(flat_list))


token_frequencies(["NLP", "NLP"])

# %%


def lemmatize_stanza(tokenlist):  # MIssing stuff
    """
    tokenlist (list): A list of tokens

    lemmatize a tokenlist using stanza
    """

    nlp = stanza.Pipeline(lang='en', processors="lemma",
                          tokenize_pretokenized=True)
    doc = nlp(tokenlist)

    res = [word.lemma for word in doc]

    return res
    

lemmatize_stanza(["These", "are", "all", "greatly", "influencial"])

# %%


def postag_stanza(tokenlist):
    """
    tokenlist (list): A list of tokens

    add a part-of-speech (POS) tag to each tokenlist using stanza
    """
    pass

# %%


class Text():
    def __init__(self, txt):
        self.sentences = sentence_segment()
        self.tokens = tokenize(self.sentences)

    def ner(self, method="regex"):
        res = ner_regex(self.tokens)
        return res

    def get_df(self):
        """
        returns a dataframe containing the columns:
        sentence number, token, lemma, pos-tag, named-entity
        """
        pass

# %%
