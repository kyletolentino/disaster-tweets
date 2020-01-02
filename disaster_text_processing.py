import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer
from string import punctuation
import re


def preprocessing(list):
    new_list = []
    for i in list:
        if i.isalpha():
            new_list.append(i)
    return new_list


tk = TweetTokenizer(strip_handles=True, reduce_len=True)
s = " What's really good @kyletolentino ? >:)    "
s2 = tk.tokenize(s)
s3 = preprocessing(s2)
print(s3)
