from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
import numpy as np
import unidecode


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


# we want stem of word, except compared to stemming, we want an actual word
def lemmatize(list):
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatize_words = np.vectorize(wordnet_lemmatizer.lemmatize)
    lemmatized_text = ' '.join(lemmatize_words(list))
    return lemmatized_text


tk = TweetTokenizer(strip_handles=True, reduce_len=True)
text = """
On the 13 Feb. 2007, Theresa May announced on MTV news that the rate of childhod obesity had 
risen from 7.3-9.6% in just 3 years , costing the N.A.T.O £20m
"""
s = tk.tokenize(text)
s1 = remove_accent(s)
s2 = no_punctuation(s1)
s3 = no_stop_words(s2)
print(lemmatize(s3))
