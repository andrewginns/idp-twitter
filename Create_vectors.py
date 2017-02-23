#!/usr/bin/env python2.7
import pandas as pd
import nltk
# nltk.download()
import re
import numpy as np
import warnings
import cPickle as pickle
import csv


from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


def review_to_words(raw_review):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        review_text = BeautifulSoup(raw_review, "html.parser").get_text()

        letters_only = re.sub("[^a-zA-Z]", " ", review_text)

        words = letters_only.lower().split()

        stops = set(stopwords.words("english"))

        meaningful_words = [w for w in words if not w in stops]

        return " ".join(meaningful_words)


"""Program Options"""
TF_IDF = 1
features = 12500 # Not applicable unless using TF_IDF = 0

pickle.dump(TF_IDF, open("setting_tfidf.p", "wb"))
pickle.dump(features, open("features.p", "wb"))

"""Create vocabulary for vectors"""
print "Creating vocabulary"
train = pd.read_csv("all.csv", header=0, delimiter="\t", quoting=csv.QUOTE_NONNUMERIC)

num_reviews = train["Tweet"].size

clean_train_reviews = []

for i in xrange(0, num_reviews):
    if(i+1) % 1000 == 0:
        print "Creating vocab from Tweet %d of %d\n" % (i+1, num_reviews)
    clean_train_reviews.append(review_to_words(train["Tweet"][i]))

pickle.dump(clean_train_reviews, open("V.p", "wb"))

if TF_IDF == 0:
    # Standard bag of words vector based on 'Bag of Words' approach
    print "Using Term Frequency (TF) for vocabulary"
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None,
                                 preprocessor=None, stop_words=None, max_features=features)
    train_data_features = vectorizer.fit_transform(clean_train_reviews).toarray()

else:
    # Implement TF-IDF weighting vectorizer
    print "Using TF-IDF weighting for vocabulary"
    vectorizer = TfidfVectorizer(min_df=1, norm=u'l1')
    train_data_features = vectorizer.fit_transform(clean_train_reviews)


"""Calculate vector for author 1"""
print "Creating author vectors"

test_h = pd.read_csv("HillaryClinton_tweets.csv", header=0, delimiter="\t", quoting=csv.QUOTE_NONNUMERIC)

num_reviews = len(test_h["Tweet"])
clean_test_h = []

for i in xrange(0, num_reviews):
    if (i + 1) % 1000 == 0:
        print "Processing Author 1, Tweet %d of %d\n" % (i + 1, num_reviews)
    clean_hill = review_to_words(test_h["Tweet"][i])
    clean_test_h.append(clean_hill)

if TF_IDF == 0:
    # Bag of words tf count
    test_hill_features = vectorizer.transform(clean_test_h).toarray()
    pickle.dump(test_hill_features, open("h_all.p", "wb"))

    h_avg = csr_matrix.mean(test_hill_features, axis=0)
    pickle.dump(h_avg, open("h.p", "wb"))

else:
    # Implement TF-IDF weighting
    train_tf_hill = vectorizer.transform(clean_test_h)
    pickle.dump(train_tf_hill, open("h_tfidf_all.p", "wb"))

    h_avg_tfidf = csr_matrix.mean(train_tf_hill, axis=0)
    pickle.dump(h_avg_tfidf, open("h_tfidf.p", "wb"))


"""Calculate vector for Author 2"""
test_t = pd.read_csv("realDonaldTrump_tweets.csv", header=0, delimiter="\t", quoting=csv.QUOTE_NONNUMERIC)

num_reviews = len(test_t["Tweet"])
clean_test_t = []

for i in xrange(0, num_reviews):
    if (i + 1) % 1000 == 0:
        print "Processing Author 2, Tweet %d of %d\n" % (i + 1, num_reviews)
    clean_trump = review_to_words(test_t["Tweet"][i])
    clean_test_t.append(clean_trump)

if TF_IDF == 0:
    # Bag of words tf count
    test_trump_features = vectorizer.transform(clean_test_t).toarray()
    pickle.dump(test_trump_features, open("t_all.p", "wb"))

    t_avg = csr_matrix.mean(test_trump_features, axis=0)
    pickle.dump(t_avg, open("t.p", "wb"))

else:
    train_tf_trump = vectorizer.transform(clean_test_t)
    pickle.dump(train_tf_trump, open("t_tfidf_all.p", "wb"))

    t_avg_tfidf = csr_matrix.mean(train_tf_trump, axis=0)
    pickle.dump(t_avg_tfidf, open("t_tfidf.p", "wb"))


"""Calculate vector for test data"""
print '\nProcessing Test Data'
a = 2
while a >= 1:
    if a == 2:
        test_t = pd.read_csv("old_realDonaldTrump_tweets.csv", header=0, delimiter="\t", quoting=csv.QUOTE_NONNUMERIC)

    if a == 1:
        test_t = pd.read_csv("old_HillaryClinton_tweets.csv", header=0, delimiter="\t", quoting=csv.QUOTE_NONNUMERIC)

    num_reviews = len(test_t["Tweet"])
    clean_test_t = []

    for i in xrange(0, num_reviews):
        if (i + 1) % 100 == 0:
            print "Processing Test Tweets, Tweet %d of %d\n" % (i + 1, num_reviews)
        clean_tweet = review_to_words(test_t["Tweet"][i])
        clean_test_t.append(clean_tweet)

    if TF_IDF == 0:
        # Bag of words tf count
        test_features = vectorizer.transform(clean_test_t)
        avg = csr_matrix.mean(test_features, axis=0)

        if a == 2:
            pickle.dump(test_features, open("oldtrump_all.p", "wb"))
            pickle.dump(avg, open("oldtrump.p", "wb"))

        if a == 1:
            pickle.dump(test_features, open("oldhill_all.p", "wb"))
            pickle.dump(avg, open("oldhill.p", "wb"))


    else:
        train_tf = vectorizer.transform(clean_test_t)
        t_avg_tfidf = csr_matrix.mean(train_tf, axis=0)

        if a == 2:
            pickle.dump(train_tf, open("oldtrump_tfidf_all.p", "wb"))
            pickle.dump(t_avg_tfidf, open("oldtrump_tfidf.p", "wb"))

        if a == 1:
            pickle.dump(train_tf, open("oldhill_tfidf_all.p", "wb"))
            pickle.dump(t_avg_tfidf, open("oldhill_tfidf.p", "wb"))

    print 'Test %d vecotors created' % a
    a -= 1

print "All selected vectors created"
