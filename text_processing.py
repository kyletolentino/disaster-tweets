from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
import numpy as np
import unidecode
import re

tw_tk = TweetTokenizer(strip_handles=True, reduce_len=True)


def preprocess_text(text):
    tokens = tw_tk.tokenize(text)
    list = remove_accent(tokens)
    list = no_punctuation(list)
    list = no_stop_words(list)
    list = remove_links(list)
    list = convert_hashtag(list)
    return lemmatize(list)


def remove_accent(list):
    no_accents = []
    for i in list:
        no_accents.append(unidecode.unidecode(i))
    return no_accents


def no_punctuation(list):
    return [i for i in list if i not in punctuation]


# "Illinois is not on fire" vs "Illinois is on fire" - should remove 'not' from stop_words
def no_stop_words(list):
    stop_words = stopwords.words('english')
    stop_words.remove('not')
    return [i for i in list if i not in stop_words]


def remove_links(list):
    no_links = []
    for i in list:
        no_links.append(re.sub(r"http\S+", "", i))
    return no_links


# not removing text with hashtag because could be important, so just stripping '#' itself
def convert_hashtag(list):
    converted = []
    for i in list:
        converted.append(re.sub(r"#", '', i))
    return converted


# we want stem of word, except compared to stemming, we want an actual word
def lemmatize(list):
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_text = ' '.join(wordnet_lemmatizer.lemmatize(i) for i in list)
    return lemmatized_text
