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

# %% TEST

test_sentence = Text("Hey guys! Karl Friston here. How is it hanging? Does this even work? Who knows!")

test_sentence.get_df()
test_sentence.tokens
test_sentence.get_ngrams(2)
# %%

## CLASS EXAMPLE: