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
    imdb_files = []
    for path, subdirs, files in os.walk("imdb"):
        for f in files:
            filepath = os.path.join(path, f)

            with open(filepath) as f:
                text = f.read()

            tag = path.split("/")[1]
            imdb_files.append([tag, text])
    return pd.DataFrame(imdb_files, columns=["tag", "text"])


if __name__ == "__main__":
    imdb = read_imdb()

    # Scikit learn
    X_train, X_test, y_train, y_test = train_test_split(imdb.text, imdb.tag)

    # Create bag-of-words
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)

    # TASK 2: What does the shape of the X_train_counts denote
    print(X_train_counts.shape)
    # hint 1 what do get if you run:
    # bow = count_vect.transform(["this is a test"])
    # bow.todense() <-- this right here transform the sparse matrix to a dense matrix
    # hint 2: what is the shape of the dense matrix?

    # Train
    clf = MultinomialNB()
    clf.fit(X_train_counts, y_train)

    # classify
    X_test_counts = count_vect.transform(
        X_test
    )  # TASK 3: Why do I use transform instead of fit_transform?
    predictions = clf.predict(X_test_counts)

    # validate
    acc = sum(predictions == y_test) / len(y_test)
    print(f"Our model obtained a performance accuracy of {acc}")

    # TASK 4: Where does the tokenization of the text happen in this script
    # and how would you change it to be your own tokenizer?
    # TASK 5 (optional): Where would you change it to use n-grams?
    # TASK 6 (optional): What about binary Naive bayes, stopword lists etc.
