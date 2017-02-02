#!/usr/bin/env python2.7
import numpy as np
import pandas as pd
import warnings
import csv
import pickle

from Tkinter import *
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial
from PIL import Image

def review_to_words( raw_review ):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        review_text = BeautifulSoup(raw_review, "html.parser").get_text()

        letters_only = re.sub("[^a-zA-Z]", " ", review_text)

        words = letters_only.lower().split()

        stops = pickle.load( open( "stopwords.p", "rb" ) )
        # stops = set(stopwords.words("english"))

        meaningful_words = [w for w in words if not w in stops]

        return ( " ".join( meaningful_words ))
def enter(event):
    master.quit()
def quit():
    master.destroy()

########################################################################################################################
""" Options
    Toggling the program between GUI and CLI.
    Clearing the text entry box after submission.
    0 for off,  1 for on.
"""
GUI = 1
clear_after_submit = 0
# TF_IDF = pickle.load( open( "setting_tfidf.p", "rb" ))
TF_IDF = 1

########################################################################################################################
""" Calculation of vector V
    Loads in V.p representing the processed text for vocabulary generation from the 'Create_vectors' program.
    Vectorizer then counts then creates a vector V to represent the top 5000 words by term frequency.
    This is then fitted to create a term-document matrix.
"""
clean_train_reviews = pickle.load( open( "V.p", "rb" ) )

if TF_IDF == 0:
# Standard bag of words vector based on 'Bag of Words' approach
    print "Using Term Frequency (TF) for vectors"
    vectorizer = CountVectorizer(analyzer= "word", tokenizer= None, preprocessor= None, stop_words= None, max_features= 5000)
    train_data_features = vectorizer.fit_transform(clean_train_reviews).toarray()

else:
# Implement TF-IDF weighting vectorizer
    print "Using TF-IDF weighting for vectors"
    vectorizer = TfidfVectorizer(min_df=1)
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

########################################################################################################################
""" Creates a GUI for the program using Tkinter
    Text widget provides an input area for tweets, assigns the input of tweets to the 'Return' key.
    Creates a 'Quit button for ending the program process.
"""
if GUI ==1:
    e = 0
    master = Tk()
    master.title("Author Recognition Tool")
    master.geometry("500x400")
    master.resizable
    Label(master, text="Enter your Tweet here:").grid(row=0)

    Label(master, text="TF-IDF status: %s" % (TF_IDF)).grid(row=1)
    e1 = Text(master)

    e1.grid(row=0, column=1)

    master.bind('<Return>', enter)
    Button(master, text='Quit', command=quit).grid(row=3, column=1, sticky=W, pady=4)

########################################################################################################################
""" Vector calculations"""
# Calculation of Input Vector
exit = 0
while exit == 0:
    if GUI ==1:
        mainloop()

    with open('Input.csv', 'wb') as csvfile:    # Creates a .csv file to store the inputted tweet.
        w = csv.writer(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
        w.writerow(["Tweet"])

        t = csv.writer(csvfile, delimiter = "\t", quoting=csv.QUOTE_ALL)
        if GUI ==1:
            input_vec = e1.get("1.0", 'end-1c')[:-1]
        else:
            input_vec = raw_input("Enter your tweet to test, type 'exit' to exit: ")

        if input_vec == "exit":
            exit = 1
        else:
            t.writerow([input_vec]) # Writes the inputted tweet to the csv file.

    if clear_after_submit == 1 and GUI ==1:
        e1.delete('1.0',END)

    test_i = pd.read_csv("Input.csv", header=0, delimiter="\t", quoting=3)
    num_reviews = len(test_i["Tweet"])
    clean_test_i = []

    for i in xrange (0, num_reviews ):
        clean_input = review_to_words( test_i["Tweet"][i])
        clean_test_i.append( clean_input )

    # Convert the new tweet into a vector after processing.
    if TF_IDF == 0:
        test_input_features = vectorizer.transform(clean_test_i).toarray()
        np.savetxt("Input_vector.csv", test_input_features, delimiter=",")
        new_vec = np.genfromtxt("Input_vector.csv", delimiter=",")

    else:
        train_tf_input = vectorizer.transform(clean_test_i)
        new_vec = np.mean(train_tf_input, axis=0)

########################################################################################################################
# Calculation of cosine similarity of new Tweet #
    if TF_IDF == 0:
        trump = pickle.load( open( "t.p", "rb" ))
        hillary = pickle.load( open( "h.p", "rb" ))
    else:
        trump = pickle.load(open("t_tfidf.p", "rb"))
        hillary = pickle.load(open("h_tfidf.p", "rb"))

    trump_sim = 1-spatial.distance.cosine(trump, new_vec)
    hill_sim = 1-spatial.distance.cosine(hillary, new_vec)

    print 'Trump cosine similarity: ',trump_sim
    print 'Hillary cosine similarity: ',hill_sim

    if trump_sim>hill_sim:
        print "You tweet like Trump"
        image = Image.open('Trump_pic.jpeg')
        image.show()

    else:
        print "You tweet like Hillary"
        image = Image.open('hill_pic.jpg')
        image.show()


