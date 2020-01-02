import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer
from nltk.stem import SnowballStemmer
import re

snowball = SnowballStemmer(language='english')


def preprocess(text):
    clean = []
    for i in (text[:][0]):
        new_text = re.sub('<.*?>', '', i)  # remove HTML tags
        new_text = re.sub(r'[^\w\s]', '', new_text)  # remove punctuation
        new_text = re.sub(r'\d+', '', new_text)  # remove numbers
        new_text = new_text.lower()  # make everything lowercase
        if new_text != '':
            clean.append(new_text)
    return clean


def stemming(words):
    stem_new = []
    for i in (words[:][0]):
        stemmed = snowball.stem(i)
        stem_new.append(stemmed)
    return stem_new


