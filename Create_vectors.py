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
from sklearn.feature_extraction.text import TfidfVectorizer

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
# Options #
# Set option = 1 if you want to create a .csv of tf-idf scores for each tweet in the corpus
option = 1
TF_IDF = 0
pickle.dump( TF_IDF, open( "setting_tfidf.p", "wb" ) )
########################################################################################################################
# Calculation of vector V #
print "Creating vocabulary"
train = pd.read_csv("realDonaldTrump_tweets.csv", header=0, delimiter="\t", quoting=3)

num_reviews = train["Tweets"].size

clean_train_reviews = []

for i in xrange (0, num_reviews ):
    if( (i+1)%1000 == 0 ):
        print "Creating vocab from Tweet %d of %d\n" % ( i+1, num_reviews )
    clean_train_reviews.append( review_to_words( train["Tweets"][i] ))

pickle.dump( clean_train_reviews, open( "V.p", "wb" ) )

if TF_IDF == 0:
# Standard bag of words vector based on 'Bag of Words' approach
    print "Using Term Frequency (TF) for vocabulary"
    vectorizer = CountVectorizer(analyzer= "word", tokenizer= None, preprocessor= None, stop_words= None, max_features= 5000)
    train_data_features = vectorizer.fit_transform(clean_train_reviews).toarray()

else:
# Implement TF-IDF weighting vectorizer
    print "Using TF-IDF weighting for vocabulary"
    vectorizer = TfidfVectorizer(min_df=1)
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

if option ==1:
########################################################################################################################
    print "Creating author vectors"
    # Calculate vector for Author 1
    test_h = pd.read_csv("HillaryClinton_tweets.csv", header=0, delimiter="\t", quoting=3)

    num_reviews = len(test_h["Tweets"])
    clean_test_h = []

    for i in xrange (0, num_reviews ):
        if( (i+1)%1000 == 0 ):
            print "Processing Author 1, Tweet %d of %d\n" % ( i+1, num_reviews )
        clean_hill = review_to_words( test_h["Tweets"][i])
        clean_test_h.append( clean_hill )

    if TF_IDF == 0:
    # Bag of words tf count
        test_hill_features = vectorizer.transform(clean_test_h).toarray()
        h_avg = np.mean(test_hill_features, axis=0)
        # np.savetxt("foo.csv", test_hill_features, delimiter=",")
        pickle.dump(h_avg, open("h.p", "wb"))

    else:
    # Implement TF-IDF weighting
        train_tf_hill = vectorizer.transform(clean_test_h)
        h_avg_tfidf = np.mean(train_tf_hill, axis=0)

        # np.savetxt("foo_idf.csv", h_avg_tfidf, delimiter=",")
        pickle.dump( h_avg_tfidf, open( "h_tfidf.p", "wb" ) )

########################################################################################################################
    # Calculate vector for Author 2
    test_t = pd.read_csv("realDonaldTrump_tweets.csv", header=0, delimiter="\t", quoting=3)

    num_reviews = len(test_t["Tweets"])
    clean_test_t = []

    for i in xrange (0, num_reviews ):
        if( (i+1)%1000 == 0 ):
            print "Processing Author 2, Tweet %d of %d\n" % ( i+1, num_reviews )
        clean_trump = review_to_words( test_t["Tweets"][i])
        clean_test_t.append( clean_trump )

    if TF_IDF == 0:
    # Bag of words tf count
        test_trump_features = vectorizer.transform(clean_test_t).toarray()
        t_avg = np.mean(test_trump_features, axis=0)
        # np.savetxt("foo2.csv", test_trump_features, delimiter=",")
        pickle.dump(t_avg, open("t.p", "wb"))

    else:
        train_tf_trump = vectorizer.transform(clean_test_t)
        t_avg_tfidf = np.mean(train_tf_trump, axis=0)

        # np.savetxt("foo2_idf.csv", t_avg_tfidf, delimiter=",")
        pickle.dump(t_avg_tfidf, open("t_tfidf.p", "wb"))

print "All selected vectors created"