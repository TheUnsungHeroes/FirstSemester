# %%
import re
import collections
import stanza

# %%
# %%


def sentence_segment(txt):
    """
    txt (str): Text which you want to be segmented into sentences.

    Example:
    >>> txt = "NLP is very cool. It is also useful"
    >>> sentence_segment(txt)
    ["NLP is very cool", "It is also useful"]
    """

    return re.findall("((?:\w+[\s]*)+)", txt)

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
    if n == 1:
        return tokenlist
    empty_list = []
    
    for i in range(len(tokenlist)-1):
        new_list = tuple(tokenlist[i:i+n])

        if len(new_list) == n:
            empty_list.append(new_list)
            
    return empty_list


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
    
    return re.findall("([A-Z][a-z]+(?:[\s]*(?:[A-Z][a-z]+))+)", tokenlist)


# %%


# def token_frequencies(tokenlist):
#     """
#     tokenlist (list): A list of tokens

#     return a list of tokens and their frequencies

#     Example:
#     >>> tokens = [["NLP", "is", "very", "cool"],
#                   ["It", "is", "also", "useful"]]
#     >>> token_frequencies(sent)
#     {"NLP": 1, "is": 2, "very": 1, "cool": 1, "It": 1, "also": 1, "useful": 1}
#     """

#     if isinstance(tokenlist[0], list):
#         flat_list = [item for sublist in tokenlist for item in sublist]
#     else:
#         flat_list = tokenlist
    
#     return {item: flat_list.count(item) for item in flat_list}


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



# %%

def lemmatize_stanza(tokenlist):  
    """
    tokenlist (list): A list of tokens

    lemmatize a tokenlist using stanza
    """
    
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma', tokenize_pretokenized=True, use_gpu = False)
    doc = nlp(tokenlist)

    return [word.lemma for sentence in doc.sentences for word in sentence.words]
    


# %%

# %%


def postag_stanza(tokenlist):
    """
    tokenlist (list): A list of tokens

    add a part-of-speech (POS) tag to each tokenlist using stanza
    """
    nlp = stanza.Pipeline(lang="en", processors="tokenize,mwt,pos", tokenize_pretokenized =True, use_gpu = False)
    doc = nlp(tokenlist)

    return [(word.text, word.pos) for sentence in doc.sentences for word in sentence.words]


# %%

def lemma_pos_stanza(tokenlist):  
    """
    tokenlist (list): A list of tokens

    lemmatize a tokenlist using stanza
    """
    
    nlp = stanza.Pipeline(lang='en', processors='mwt,tokenize,pos,lemma', tokenize_pretokenized=True, use_gpu = False)
    doc = nlp(tokenlist)

    return [(sentence_no+1, word.id, word.text, word.lemma, word.pos) for sentence_no, sentence in enumerate(doc.sentences) for word in sentence.words]
    

# %%
# %%



def lemma_pos_stanza(tokenlist):  
    """
    tokenlist (list): A list of tokens

    lemmatize a tokenlist using stanza
    """
    
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma', tokenize_pretokenized=True, use_gpu = False)
    doc = nlp(tokenlist)

    return [(sentence_number+1, word.id, word.text, word.lemma, word.pos) for sentence_number, sentence in enumerate(doc.sentences) for word in sentence.words]
    
    
class Text():
    '''
    This class contains the entire pipeline of pre-processing
    of text using the text_processor_functions and stanza.
    '''
    def __init__(self, txt):

        '''Initializing the class. This includes:
        1. sentences = the sentences of the text in a list.
        2. tokens = the tokens of the sentences in a list of lists. '''

        self.sentences = sentence_segment(txt)
        self.tokens = tokenize(self.sentences)

    def ner(self, method="regex"):

        '''
        Returns a list of lists containing the tokens
        of the named entities in the text.
        '''

        res = list(map(ner_regex, self.sentences))
        names = [name for name in res if name]
        names = tokenize(names[0])[0]

        return names

    def get_df(self):
        """
        returns a dataframe containing the columns:
        sentence number, token, lemma, pos-tag, named-entity
        """
        tokens = self.tokens

        names = test_sentence.ner()

        names_list = []
        for sentence in tokens:
            for word in sentence:
                names_list.append(word in names)

        df = pd.DataFrame(lemma_pos_stanza(tokens), columns = ["Sent_No", "Word_No","Word", "Lemma", "POS"])
        df["NE"] = names_list
        return df

    def get_ngrams(self, n):
        """
        returns n-grams on text
        """
        return [n_grams(token, n) for token in self.tokens]


























# %%
