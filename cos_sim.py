#!/usr/bin/env python2.7
import numpy as np
import pandas as pd
import warnings
import csv
import cPickle as pickle
import math
import time

from Tkinter import *
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial
from PIL import Image


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


def enter(event):
    master.quit()


def stop():
    master.destroy()


def percentage(nom, denom):
    return 100*float(nom)/float(denom)


def weighting(tfidf, num):
    if tfidf == 0:
        print "Using Top %d by Term Frequency (TF) for vectors" % num
    else:
        print "Using TF-IDF weighting for vectors"
    return
########################################################################################################################
""" Options
    Toggling the program between GUI and CLI.
    Clearing the text entry box after submission.
    Automation toggle.
    0 for off,  1 for on.
"""
GUI = 1
clear_after_submit = 1
TF_IDF = pickle.load( open( "setting_tfidf.p", "rb" ))
# TF_IDF = 1
features = pickle.load( open( "features.p", "rb" ))
automate = 0

########################################################################################################################
""" Calculation of vector V
    Loads in V.p representing the processed text for vocabulary generation from the 'Create_vectors' program.
    Vectorizer then counts then creates a vector V to represent the top 5000 words by term frequency.
    This is then fitted to create a term-document matrix.
"""
clean_train_reviews = pickle.load(open("V.p", "rb"))

if TF_IDF == 0:
    # Standard bag of words vector based on 'Bag of Words' approach
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=features)
    train_data_features = vectorizer.fit_transform(clean_train_reviews).toarray()

else:
    # Implement TF-IDF weighting vectorizer
    vectorizer = TfidfVectorizer(min_df=1)
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

########################################################################################################################
""" Creates a GUI for the program using Tkinter
    Text widget provides an input area for tweets, assigns the input of tweets to the 'Return' key.
    Creates a 'Quit button for ending the program process.
"""
if GUI == 1:
    master = Tk()
    master.title("Author Recognition Tool - Andrew Ginns IDP")
    master.geometry("720x400")

    Label(master, text="Enter your Tweet here:").grid(row=0)
    Label(master, text="TF-IDF status: %s" % TF_IDF).grid(row=1)

    e1 = Text(master)
    e1.grid(row=0, column=1)

    master.bind('<Return>', enter)
    Button(master, text='Quit', command=stop).grid(row=3, column=1, sticky=W, pady=4)

########################################################################################################################
""" Vector calculations
    Input vectors calculated and compared against vectors representing authors.
    Output based on the author with the most similarity to the input.
"""
# Calculation of Input Vector #
end = 0
while end == 0 and automate == 0:
    if GUI == 1:
        mainloop()

    with open('Input.csv', 'wb') as csvfile:    # Creates a .csv file to store the inputted tweet.
        w = csv.writer(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
        w.writerow(["Tweet"])

        t = csv.writer(csvfile, delimiter="\t", quoting=csv.QUOTE_ALL)
        if GUI == 1:
            input_vec = e1.get("1.0", 'end-1c')[:-1]

        else:
            input_vec = raw_input("Enter your tweet to test, type 'exit' to exit: ")

        if input_vec == "exit":
            end = 1
        else:
            t.writerow([input_vec])  # Writes the inputted tweet to the csv file.

    if clear_after_submit == 1 and GUI == 1:
        e1.delete('1.0', END)

    test_i = pd.read_csv("Input.csv", header=0, delimiter="\t", quoting=3)
    num_reviews = len(test_i["Tweet"])
    clean_test_i = []

    for i in xrange(0, num_reviews):
        clean_input = review_to_words(test_i["Tweet"][i])
        clean_test_i.append(clean_input)

    # Convert the new tweet into a vector after processing.
    if TF_IDF == 0:
        test_input_features = vectorizer.transform(clean_test_i).toarray()
        np.savetxt("Input_vector.csv", test_input_features, delimiter=",")
        new_vec = np.genfromtxt("Input_vector.csv", delimiter=",")

    else:
        train_tf_input = vectorizer.transform(clean_test_i)
        new_vec = np.mean(train_tf_input, axis=0)


# Classifier using cosine similarity of new Tweet #
    if TF_IDF == 0:
        trump = pickle.load(open("t.p", "rb"))
        hillary = pickle.load(open("h.p", "rb"))

    else:
        trump = pickle.load(open("t_tfidf.p", "rb"))
        hillary = pickle.load(open("h_tfidf.p", "rb"))

    trump_sim = 1-spatial.distance.cosine(trump, new_vec)
    hill_sim = 1-spatial.distance.cosine(hillary, new_vec)

    print 'Trump cosine similarity: ', trump_sim
    print 'Hillary cosine similarity: ', hill_sim

    if math.isnan(trump_sim):
        print "No similarity!"
        image = Image.open('Broken.jpg')
        image.show()

    if trump_sim > hill_sim:
        print "You tweet like Trump"
        image = Image.open('Trump_pic.jpeg')
        image.show()

    if hill_sim > trump_sim:
        print "You tweet like Hillary"
        image = Image.open('hill_pic.jpg')
        image.show()

########################################################################################################################
""" Automation
    Allows evaluation of the program by providing percentage metrics of tweets associated to each author.
    Enables processing of a large number of tweets for author classification.
"""
if automate == 1:
    a = 3
    while a >=1:

        u = 0
        t = 0
        h = 0

        if a == 3:
            test_i = pd.read_csv("new_realDonaldTrump_tweets.csv", header=0, delimiter="\t", quoting=3, encoding='utf8')
            print '\n Testing for Trump Tweets'

        if a == 2:
            test_i = pd.read_csv("older_realDonaldTrump_tweets.csv", header=0, delimiter="\t", quoting=3, encoding='utf8')
            print '\n Testing for old Trump Tweets'

        if a == 1:
            test_i = pd.read_csv("new_HillaryClinton_tweets.csv", header=0, delimiter="\t", quoting=3, encoding='utf8')
            print '\n Testing for Hillary Tweets'

        num_reviews = len(test_i["Tweet"])

        print "Tweets to process: %d" % num_reviews

        for i in xrange(0, num_reviews):

            clean_input = review_to_words(test_i["Tweet"][i])

            with open('Input.csv', 'wb') as csvfile:  # Creates a .csv file to store the inputted tweet.
                w = csv.writer(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
                w.writerow(["Tweet"])
                w.writerow(clean_input)

            clean_test_i = [clean_input]

            # Convert the new tweet into a vector after processing.
            if TF_IDF == 0:
                test_input_features = vectorizer.transform(clean_test_i).toarray()
                np.savetxt("Input_vector.csv", test_input_features, delimiter=",")
                new_vec = np.genfromtxt("Input_vector.csv", delimiter=",")

            else:
                train_tf_input = vectorizer.transform(clean_test_i)
                new_vec = np.mean(train_tf_input, axis=0)

    # Comparing cosine similarity
            if TF_IDF == 0:
                trump = pickle.load(open("t.p", "rb"))
                hillary = pickle.load(open("h.p", "rb"))

            else:
                trump = pickle.load(open("t_tfidf.p", "rb"))
                hillary = pickle.load(open("h_tfidf.p", "rb"))

            trump_sim = 1 - spatial.distance.cosine(trump, new_vec)
            hill_sim = 1 - spatial.distance.cosine(hillary, new_vec)

            if math.isnan(trump_sim):
                u += 1

            if trump_sim > hill_sim:
                t += 1

            if hill_sim > trump_sim:
                h += 1

        total = u + t + h
        weighting(TF_IDF, features)
        print "Number unclassified = %d, %d percent" % (u, percentage(u, total))
        print "Number of Trump tweets = %d, %d percent" % (t, percentage(t, total))
        print "Number of Hillary tweets = %d, %d percent" % (h, percentage(h, total))
        a -= 1
