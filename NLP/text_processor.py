""" LASSE go to facebook
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
# %%
import re
import collections
import stanza
import pandas as pd 
from text_processor_functions import * 


# %%
# %%

class Text():
    def __init__(self, txt):
        self.sentences = sentence_segment(txt)
        self.tokens = tokenize(self.sentences)

    def ner(self, method="regex"):
        res = list(map(ner_regex, self.sentences))
        return res

    def get_df(self):
        """
        returns a dataframe containing the columns:
        sentence number, token, lemma, pos-tag, named-entity
        """
        tokens = self.tokens

        return pd.DataFrame(lemma_pos_stanza(tokens), columns = ["Sent_No", "Word_No","Word", "Lemma", "POS"])

# %% TEST

test_sentence = Text("Hey guys! How is it hanging? Does this even work? Who knows!")

test_sentence.tokens

test_sentence.get_df()
# %%