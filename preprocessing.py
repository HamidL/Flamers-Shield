from nltk import TweetTokenizer, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import swifter
import re

test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

tokenizer = TweetTokenizer()
wnl = WordNetLemmatizer()


def raw(text):
    return text


def lowercase(text):
    return text.lower()


def whitespace(text):
    return " ".join(text.split())


def punctuations(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def stopwords(text):
    text = [w for w in text.split() if w not in set(stopwords)]  # remove stopwords
    return " ".join(text)


def stemming(text):
    res = []
    text = word_tokenize(text)  # stemming/lemmatization (remove sparse words)
    for word in text:
        res.append(stemmer.stem(word))  # stemming
    return " ".join(res)


def lemmatize(text):
    res = []
    text = word_tokenize(text)  # stemming/lemmatization (remove sparse words)
    for word in text:
        res.append(lemmatizer.lemmatize(word))  # lemmatizer
    return " ".join(res)


def P1(text):
    text = text.lower()
    text = " ".join(text.split())
    return text.translate(str.maketrans("", "", string.punctuation))


def P2(text):
    text = text.lower()
    text = " ".join(text.split())
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = [w for w in text.split() if w not in set(stopwords)]
    return " ".join(text)


def P3(text):
    text = text.lower()
    text = " ".join(text.split())
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = [w for w in text.split() if w not in set(stopwords)]
    text = " ".join(text)
    text = word_tokenize(text)
    res = []
    for word in text:
        res.append(stemmer.stem(word))
    return " ".join(res)


def P4(text):
    text = text.lower()
    text = " ".join(text.split())
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = [w for w in text.split() if w not in set(stopwords)]
    text = " ".join(text)
    text = word_tokenize(text)
    res = []
    for word in text:
        res.append(lemmatizer.lemmatize(word))
    return " ".join(res)


import time
start = time.time()
test["comment_text"] = test["comment_text"].swifter.apply(P4)
end = time.time()
print(end - start)

test.to_csv("test_cleaned.csv",index=False)