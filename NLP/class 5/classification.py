"""
A script for classifying IMDB movies reviews into positive and negative
categories.

What you need to do in class:
1) skim the script, get an overview of the functions as the structure of the
script
2) search for 'TASK' and solve each of these tasks (starting from the top)
3) Try out different classifiers can you achieve a better performance than you
did in the assignment? Hint: check of the script classification_additional.ipynb
"""

import os

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def read_imdb():
    """
    TASK 1: Write a short description about what this function does and how it
    does it. You should also describe the structure of imdb_files.
    """

    '''
    This function uses the 'os'-package to find the files and the directories 
    containing the imdb-data. The directories and the paths are saved, in order
    to read the different files. This also ensures that one can obtain the
    correct tag of a file using its location in the directory as a proxy.
    Thus the function assumes that all negative files are in one directory and
    all positives are in another directory. 
    ________
    Input: None
    Returns Pandas Dataframe containing the columns tag, corresponding to
    the correct label for the text, and the text. 
    '''

    imdb_files = []
    for path, subdirs, files in os.walk("imdb"):
        for f in files:
            filepath = os.path.join(path, f)

            with open(filepath) as f:
                text = f.read()
            
            tag = os.path.split(path)[1]
            imdb_files.append([tag, text])
    return pd.DataFrame(imdb_files, columns=["tag", "text"])


if __name__ == "__main__":
    imdb = read_imdb()

    # Scikit learn
    X_train, X_test, y_train, y_test = train_test_split(imdb.text, imdb.tag)

    # Create bag-of-words
    count_vect = CountVectorizer(analyzer='word', binary=True, ngram_range=(1, 2))
    X_train_counts = count_vect.fit_transform(X_train)

    # TASK 2: What does the shape of the X_train_counts denote
    '''
    The first number in the tuple is the amount of documents
    in the training data. The second denotes the number of 
    words in the training data. NOTE: only takes words
    that are in the training data. Removes single letters and digits(?).
    '''
    print(X_train_counts.shape)
    # hint 1 what do get if you run:
    #bow = count_vect.transform(["an","be"]) #where in our vector are the different words represented? And how many times 
    #bow.todense() #<-- this right here transform the sparse matrix to a dense matrix
    #print(bow.shape)
    #print(bow)
    #len(X_train)
    # hint 2: what is the shape of the dense matrix?

    # Train
    clf = MultinomialNB()
    clf.fit(X_train_counts, y_train)

    # classify
    X_test_counts = count_vect.transform(
        X_test
    )  # TASK 3: Why do I use transform instead of fit_transform?
    '''
    We already fitted the data - and we are not interested in 
    fitting our model to the test data. We need to test using this 
    data. So instead, we need only to transform the counts into
    the format that the fitted data is in. This is what transform 
    does. 
    '''
    predictions = clf.predict(X_test_counts)

    # validate
    acc = sum(predictions == y_test) / len(y_test)
    print(f"Our model obtained a performance accuracy of {acc}")

    # TASK 4: Where does the tokenization of the text happen in this script
    # and how would you change it to be your own tokenizer?

    '''
    Happens in count vectorizer - does pretty much the whole course until now.
    '''
    # TASK 5 (optional): Where would you change it to use n-grams?
    '''
    You have to extend the CountVectorizer function to include ngrams
    '''
    # TASK 6 (optional): What about binary Naive bayes, stopword lists etc.


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)

# Scikit learn
X_train, X_test, y_train, y_test = train_test_split(imdb.text, imdb.tag)

# Create bag-of-words
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_test_counts = count_vect.transform(X_test)

clf.fit(X_train_counts, y_train_test)
predictions = clf.predict(X_test_counts)

acc = sum(predictions == y_test) / len(y_test)
print(f"Our model obtained a performance accuracy of {acc}")