import pandas as pd
import nltk
# nltk.download()
import re
import bs4
import sklearn
import numpy as np
import warnings
import pickle

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def review_to_words( raw_review ):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        review_text = BeautifulSoup(raw_review, "html.parser").get_text()

        letters_only = re.sub("[^a-zA-Z]", " ", review_text)

        words = letters_only.lower().split()

        stops = set(stopwords.words("english"))

        meaningful_words = [w for w in words if not w in stops]

        return ( " ".join( meaningful_words ))

########################################################################################################################
# Calculation of vector V #
train = pd.read_csv("realDonaldTrump_tweets.csv", header=0, delimiter="\t", quoting=3)

num_reviews = train["Tweets"].size

clean_train_reviews = []

for i in xrange (0, num_reviews ):
    if( (i+1)%1000 == 0 ):
        print "Tweet %d of %d\n" % ( i+1, num_reviews )
    clean_train_reviews.append( review_to_words( train["Tweets"][i] ))

pickle.dump( clean_train_reviews, open( "V.p", "wb" ) )
## Replace clean_train_reviews with words with large usefulness??

vectorizer = CountVectorizer(analyzer= "word", tokenizer= None, preprocessor= None, stop_words= None, max_features= 5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews).toarray()

# Implement TF-IDF weighting
tf_transformer = TfidfTransformer(use_idf=False).fit(train_data_features)
train_tf = tf_transformer.transform(train_data_features)

########################################################################################################################
# Calculate vector for Author 1
test_h = pd.read_csv("HillaryClinton_tweets.csv", header=0, delimiter="\t", quoting=3)

num_reviews = len(test_h["Tweets"])
clean_test_h = []

for i in xrange (0, num_reviews ):
    if( (i+1)%1000 == 0 ):
        print "Tweet %d of %d\n" % ( i+1, num_reviews )
    clean_hill = review_to_words( test_h["Tweets"][i])
    clean_test_h.append( clean_hill )

test_hill_features = vectorizer.transform(clean_test_h).toarray()

# Implement TF-IDF weighting
tf_transformer_hill = TfidfTransformer(use_idf=False).fit(test_hill_features)
train_tf_hill = tf_transformer.transform(test_hill_features)

np.savetxt("foo.csv", test_hill_features, delimiter=",")

########################################################################################################################
# Calculate vector for Author 2
test_t = pd.read_csv("realDonaldTrump_tweets.csv", header=0, delimiter="\t", quoting=3)

num_reviews = len(test_t["Tweets"])
clean_test_t = []

for i in xrange (0, num_reviews ):
    if( (i+1)%1000 == 0 ):
        print "Tweet %d of %d\n" % ( i+1, num_reviews )
    clean_trump = review_to_words( test_t["Tweets"][i])
    clean_test_t.append( clean_trump )

test_trump_features = vectorizer.transform(clean_test_t).toarray()

# Implement TF-IDF weighting
tf_transformer_trump = TfidfTransformer(use_idf=False).fit(test_trump_features)
train_tf_trump = tf_transformer.transform(test_trump_features)


np.savetxt("foo2.csv", test_trump_features, delimiter=",")

