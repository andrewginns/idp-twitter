import pandas as pd
import nltk
# nltk.download()
import re
import bs4
import sklearn
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

def review_to_words( raw_review ):
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()

    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    words = letters_only.lower().split()

    stops = set(stopwords.words("english"))

    meaningful_words = [w for w in words if not w in stops]

    return ( " ".join( meaningful_words ))


train = pd.read_csv("realDonaldTrump_tweets.csv", header=0, delimiter="\t", quoting=3)

print train.shape

print train.columns.values

# print train["Tweets"][0]

# clean_review = review_to_words( train["Tweets"][0])
# print clean_review

num_reviews = train["Tweets"].size

clean_train_reviews = []

for i in xrange (0, num_reviews ):
    if( (i+1)%1000 == 0 ):
        print "Tweet %d of %d\n" % ( i+1, num_reviews )
    clean_train_reviews.append( review_to_words( train["Tweets"][i] ))

## Replace clean_train_reviews with words with large usefulness??

vectorizer = CountVectorizer(analyzer= "word", tokenizer= None, preprocessor= None, stop_words= None, max_features= 5000)

train_data_features = vectorizer.fit_transform(clean_train_reviews)

train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()

########################################################################################################################
# Calculate vector for Author 1
test_h = pd.read_csv("HillaryClinton_tweets.csv", header=0, delimiter="\t", quoting=3)

print test_h.shape
print test_h.columns.values

num_reviews = len(test_h["Tweets"])
clean_test_h = []

for i in xrange (0, num_reviews ):
    if( (i+1)%1000 == 0 ):
        print "Tweet %d of %d\n" % ( i+1, num_reviews )
    clean_hill = review_to_words( test_h["Tweets"][i])
    clean_test_h.append( clean_hill )

test_hill_features = vectorizer.transform(clean_test_h)
test_hill_features = test_hill_features.toarray()

np.savetxt("foo.csv", test_hill_features, delimiter=",")

########################################################################################################################
# Calculate vector for Author 2
test_t = pd.read_csv("realDonaldTrump_tweets.csv", header=0, delimiter="\t", quoting=3)

print test_t.shape
print test_t.columns.values

num_reviews = len(test_t["Tweets"])
clean_test_t = []

for i in xrange (0, num_reviews ):
    if( (i+1)%1000 == 0 ):
        print "Tweet %d of %d\n" % ( i+1, num_reviews )
    clean_trump = review_to_words( test_t["Tweets"][i])
    clean_test_t.append( clean_trump )

test_trump_features = vectorizer.transform(clean_test_t)
test_trump_features = test_trump_features.toarray()

np.savetxt("foo2.csv", test_trump_features, delimiter=",")

