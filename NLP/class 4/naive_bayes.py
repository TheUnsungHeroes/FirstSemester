"""
prepping for class
"""
from collections import Counter

import pandas as pd
import numpy as np

from text_preprocessor import Text


def train_test_split(df, test_pct=0.2, downsample=True):
    if downsample:
        min_sam = min([sum(df["category"] == "spam"), sum(df["category"] == "ham")])
        tmp = df[df["category"] == "spam"]
        df = tmp.append(df[df["category"] == "ham"][:min_sam])

    # create mask
    mask = np.random.choice([0, 1], p=[1-test_pct, test_pct],
                            size=df.shape[0])

    # apply mask
    df["mask"] = mask
    test_df = df[df["mask"] == 1]
    train_df = df[df["mask"] == 0]

    # remove mask col
    test_df = test_df.drop("mask", axis="columns").reset_index()
    train_df = train_df.drop("mask", axis="columns").reset_index()

    return train_df, test_df


def read_data(test_split=0.8):
    df = pd.read_csv("spam.csv", encoding="latin-1")
    # df = pd.read_csv("test.csv", encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["category", "text"]

    if test_split:
        train, test = train_test_split(df, test_pct=0.2)
        return train, test
    return df


def get_frec(txt):
    t = Text(txt)
    return t.get_frequencies()


def remove_counts_less_than(counter, n=2):
    return Counter({k: count_ham[k] for k in count_ham if count_ham[k] > n-1}) 


def get_counts(df):
    count_spam = Counter()
    count_ham = Counter()
    for idx, row in df.iterrows():
        if row.category == "spam":
            count_spam += get_frec(row["text"])
        else:
            count_ham += get_frec(row["text"])
    return count_ham, count_spam


def classify(text):
    t = Text(text)
    counts = t.get_frequencies()

    is_spam = p_spam
    for k in counts:
        if k in likelihood_spam:
            is_spam = is_spam + likelihood_spam[k]

    is_ham = p_ham
    for k in counts:
        if k in likelihood_ham:
            is_ham = is_ham + likelihood_ham[k]

    if is_spam < is_ham:
        return "spam"
    return "ham"


if __name__ == "__main__":
    train, test = read_data()

    # inspect data
    # sample = test["text"][19]
    # cat = train["category"][300]

    # t = Text(sample)
    # counts = t.get_frequencies()
    # print(f"{cat} - {sample}\n Text counts \n\n{counts}")

    # create counts
    # ...for train
    count_ham, count_spam = get_counts(train)

    # remove singular words
    print(len(count_ham.keys()))
    count_ham = remove_counts_less_than(count_ham, n=2)
    print(len(count_ham.keys()))

    # calculate prior of spam
    p_spam = sum(train["category"] == "spam")/train.shape[0]
    p_ham = np.log(1-p_spam)
    p_spam = np.log(p_spam)
    print(p_spam)
    print(p_ham)

    # calculate p(w|c) probability of word given condition
    arr = np.array(list(count_spam.values()))
    likelihood = np.log((arr + 1)/(sum(arr) + len(arr)))  # use add 1 smoothing
    likelihood_spam = dict(zip(count_spam.keys(), likelihood))

    arr = np.array(list(count_ham.values()))
    likelihood = np.log((arr + 1)/(sum(arr) + len(arr)))  # use add 1 smoothing
    likelihood_ham = dict(zip(count_ham.keys(), likelihood))


    # classify the test
    # is_spam = p_spam
    # for k in counts:
    #     if k in likelihood:
    #         is_spam = is_spam + likelihood_spam[k]

    # is_ham = p_ham
    # for k in counts:
    #     if k in likelihood:
    #         is_ham = is_ham + likelihood_ham[k]

    # print(f" is it spam? {(is_spam > is_ham)}")

    test["pred"] = test["text"].apply(classify)
    print(sum(test["pred"] == test["category"])/test.shape[0])



# # Examples of how to use numpy to simplify code
# # calculation of log likelihood for a category
# bow = np.array([1, 2, 0, 0, 1, 1])
# ll = np.log((bow + 1)/(sum(bow) + len(bow))

# # large scale classification
# bow = np.array([[2, 1, 0.1], [0, 0, 1]])
# bow += 1  # laplace smoothing
# ll = np.array([0.01, 1, 0.2])  # log likelihood vector
# lp = np.array([0.5, 0.5])  # log prior vector

# # calculation of p(w_i|c) for multiple docs at once
# np.prod(bow*ll, axis=1) + lp
