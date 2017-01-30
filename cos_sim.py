import numpy as np
import pandas as pd
import re
import warnings
import csv
import pickle

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
def run():
    master.quit()
def quit():
    master.destroy()


########################################################################################################################
# Calculation of vector V #
clean_train_reviews = pickle.load( open( "V.p", "rb" ) )

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
# GUI creation #
master = Tk()
master.title("Author Recognition Tool")
master.geometry("500x400")
master.resizable
Label(master, text="Enter your Tweet here:").grid(row=0)

e1 = Text(master)

e1.grid(row=0, column=1)

Button(master, text='Analyse', command=run).grid(row=3, column=0, sticky=W, pady=4)
Button(master, text='Quit', command=quit).grid(row=3, column=1, sticky=W, pady=4)

########################################################################################################################
# Calculation of input vector #
exit = 0
while exit == 0:
    mainloop()

    with open('Input.csv', 'wb') as csvfile:
        w = csv.writer(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
        w.writerow(["Tweet"])

        t = csv.writer(csvfile, delimiter = "\t", quoting=csv.QUOTE_ALL)
        input_vec = e1.get("1.0", 'end-1c')
        # input_vec = raw_input("Enter your tweet to test, type 'enter' to exit: ")
        if input_vec == "exit":
            exit = 1
        else:
            t.writerow([input_vec])

    test_i = pd.read_csv("Input.csv", header=0, delimiter="\t", quoting=3)
    num_reviews = len(test_i["Tweet"])
    clean_test_i = []

    for i in xrange (0, num_reviews ):
        clean_input = review_to_words( test_i["Tweet"][i])
        clean_test_i.append( clean_input )

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
