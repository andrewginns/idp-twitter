#!/usr/bin/env python2.7
import cPickle as pickle
import operator
import math
import pandas as pd
import warnings
import csv
import os

from Tkinter import *
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy import spatial
from PIL import Image


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def review_to_words(raw_review):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        review_text = BeautifulSoup(raw_review, "html.parser").get_text()

        letters_only = re.sub("[^a-zA-Z]", " ", review_text)

        words = letters_only.lower().split()

        stops = pickle.load(open("stopwords.p", "rb"))
        # stops = set(stopwords.words("english"))

        meaningful_words = [w for w in words if not w in stops]

        return " ".join(meaningful_words)


GUI = 0
k = 551
dimensions = 50

"""Checking parameters are valid"""
path = '%s/%d' % (os.getcwd(), dimensions)
if os.path.exists(path) is True:
    print '\nDimension folder found'
else:
    print '\nNo folder found for %d dimensions, please run Create_SVD with required dimensions first' % dimensions
    exit()

n = pickle.load(open("%s/num_dimen.p" % path, "rb"))

if str(n) == str(dimensions):
    print '%d SVD dimensions loaded' % dimensions

else:
    print 'This path contains %d dimensions, please run Create_SVD with required dimensions first' % n
    exit()

"""Loading trained SVD"""
svd = pickle.load(open("%s/svd_trained.p" % path, "rb"))

"""Loading training data"""
train_data = pickle.load(open("%s/labelled_svd_train.p" % path, "rb"))
print '\nTraining data shape:'
print train_data.shape
print '\n Trump Train ID: %d' % train_data[0][dimensions]
print '\n Hillary Train ID: %d' % train_data[-1][dimensions]


"""TF-IDF vectorizer for input queries"""
clean_train_reviews = pickle.load(open("V.p", "rb"))
vectorizer = TfidfVectorizer(min_df=1)
train_data_features = vectorizer.fit_transform(clean_train_reviews)

"""Loading test data"""
end = 0
print '\nK size: %d' % k
print 'SVD dimensions: %d' % n
while end == 0:
    print 'Waiting for query data'
    clean_test_i = []

    with open('Input.csv', 'wb') as csvfile:  # Creates a .csv file to store the inputted tweet.
        w = csv.writer(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
        w.writerow(["Tweet"])

        t = csv.writer(csvfile, delimiter="\t", quoting=csv.QUOTE_ALL)

        if GUI == 1:
            print 'No GUI yet!'
            # input_vec = e1.get("1.0", 'end-1c')[:-1]

        else:
            input_vec = raw_input("Enter your tweet to test, type 'exit' to exit: ")

        if input_vec == "exit":
            end = 1

        else:
            t.writerow([input_vec])  # Writes the inputted tweet to the csv file.

    # if clear_after_submit == 1 and GUI == 1:
    #     e1.delete('1.0', END)

    test_i = pd.read_csv("Input.csv", header=0, delimiter="\t", quoting=3)
    num_reviews = len(test_i["Tweet"])

    for i in xrange(0, num_reviews):
        clean_input = review_to_words(test_i["Tweet"][i])
        clean_test_i.append(clean_input)

    train_tf_input = vectorizer.transform(clean_test_i)
    input_vector = train_tf_input.toarray()
    print '\nInput data shape'
    print input_vector.shape
    input = svd.transform(input_vector)
    print '\nTransformed input data shape'
    print input.shape

    """Testing accuracy for different values of k"""

    neighbors = getNeighbors(train_data, input[0], k)
    result = getResponse(neighbors)

    if result == 1.0:
        print "You tweet like @realDonaldTrump"

    else:
        print "You tweet like @HillaryClinton"
