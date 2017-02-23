import csv
import cPickle as pickle
import numpy as np
import operator
import math
import os


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
            # print 'Match'
    return (correct / float(len(testSet))) * 100.0

"""Choose number of nearest neighbours and dimensions to test"""
for dimensions in range(100, 101, 50):
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

    """Training data"""
    train_data = pickle.load(open("%s/labelled_svd_train.p" % path, "rb"))
    print '\nTraining data shape:'
    print train_data.shape
    print '\n Trump Train ID: %d' % train_data[0][dimensions]
    print '\n Hillary Train ID: %d' % train_data[-1][dimensions]

    """Test data"""
    # Old Trump
    test_data1 = pickle.load(open("%s/oldT_lsa_labelled.p" % path, "rb"))

    # Old Hillary
    test_data2 = pickle.load(open("%s/oldH_lsa_labelled.p" % path, "rb"))

    test_data = np.append(test_data1, test_data2, 0)
    print '\n Trump Test ID: %d' % test_data[0][dimensions]
    print '\n Hillary Test ID: %d' % test_data[-1][dimensions]

    if test_data[0][dimensions] == train_data[0][dimensions] and test_data[-1][dimensions] == train_data[-1][dimensions]:
        print '\nTraining data and test data author IDs match'

    else:
        print '\nError, IDs do not match'
        exit()


    print '\nCombined Test Size'
    print test_data.shape
    if len(test_data[0]) == len(train_data[0]): # Checking that the dimensions of both are equal when labelled
        print '\nTraining data and test data loaded correctly'
        print type(train_data)
        print type(test_data)

    else:
        print 'Error'
        exit()


    """Running the program"""

    for k in range(1, 652, 50):
        predictions = []

        print '\nRunning the k-nn algorithm'
        print 'Number of dimensions: %d' % n
        print 'K size: %d\n' % k
        for x in range(len(test_data)):
            if (x+1) % 1000 == 0:
                print "Processing test tweets, tweet %d of %d\n" % (x+1, len(test_data))

            neighbors = getNeighbors(train_data, test_data[x], k)
            result = getResponse(neighbors)
            predictions.append(result)
            # print('> predicted=' + repr(result) + ', actual=' + repr(test_data[x][-1]))

        accuracy = getAccuracy(test_data, predictions)
        with open('%s/%d_dimen_%d_nn_Result.csv' % (path, n, k), 'wb') as csvfile:  # Creates .csv file to store the inputted tweet.
            w = csv.writer(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
            w.writerow(['Accuracy:%s, Dimensions:%d, k size:%d' % (repr(accuracy), n, k)])
        print('Accuracy: ' + repr(accuracy) + '%')
        print 'K size: %d' % k
