import numpy as np
import pandas as pd
import re
import warnings
import csv

from Tkinter import *
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy import spatial
from PIL import Image

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
# train = pd.read_csv("HillaryClinton_tweets.csv", header=0, delimiter="\t", quoting=3)

num_reviews = train["Tweets"].size

clean_train_reviews = []

for i in xrange (0, num_reviews ):
    clean_train_reviews.append( review_to_words( train["Tweets"][i] ))

vectorizer = CountVectorizer(analyzer= "word", tokenizer= None, preprocessor= None, stop_words= None, max_features= 5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews).toarray

# Vocab from a pre-calculated list #
# with open('Vocab_Trump.csv') as f:
#     vocab = dict(filter(None, csv.reader(f)))
# # print vocab.keys()
#
# vectorizer = DictVectorizer()
# vectorizer.fit_transform(vocab).toarray()

# print vectorizer.get_feature_names()

########################################################################################################################
# Calculation of input vector #
test_i = pd.read_csv("Input.csv", header=0, delimiter="\t", quoting=3)

num_reviews = len(test_i["Tweet"])
clean_test_i = []

for i in xrange (0, num_reviews ):
    clean_hill = review_to_words( test_i["Tweet"][i])
    clean_test_i.append( clean_hill )

test_input_features = vectorizer.transform(clean_test_i)
test_input_features = test_input_features.toarray()

np.savetxt("Input_vector.csv", test_input_features, delimiter=",")
new_vec = np.genfromtxt("Input_vector.csv", delimiter=",")

########################################################################################################################
# Calculation of cosine similarity of new Tweet #
trump = np.genfromtxt("trump_tf_idf.csv", delimiter=",")
hillary = np.genfromtxt("hill_tf_idf.csv", delimiter=",")

trump_sim = 1-spatial.distance.cosine(trump, new_vec)
hill_sim = 1-spatial.distance.cosine(hillary, new_vec)

print trump_sim
print hill_sim

if trump_sim>hill_sim:
    print "You tweet like Trump"
    image = Image.open('Trump_pic.jpeg')
    image.show()

else:
    print "You tweet like Hillary"
    image = Image.open('hill_pic.jpg')
    image.show()
