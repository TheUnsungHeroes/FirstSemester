"""
This script is just intended as a placeholder for your own text preprocessing
script
"""

from tensorflow.keras.preprocessing.text import text_to_word_sequence
from collections import Counter


class Text():
    def __init__(self, txt):
        self.tokens = text_to_word_sequence(txt,
                                            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                            lower=False,
                                            split=' ')

    def get_frequencies(self):
        return Counter(self.tokens)

