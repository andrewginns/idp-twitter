#!/usr/bin/env python2.7
import cPickle as pickle
import numpy as np
import os
import csv

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

"""Choose number of SVD dimensions to use"""
for n in range(80, 101, 20):
    path = os.getcwd()
    destination = '%s/%d' % (path, n)
    assert os.path.exists(path)
    os.makedirs(destination)
    print '\n\n%d dimensions' % n
    print '\nFolder for SVD files created'

    """Loading Trump Data"""
    trump = pickle.load(open("t_tfidf_all.p", "rb"))

    tr = trump.toarray()
    print '\nTrump data shape'
    print tr.shape

    new_col1 = tr.sum(1)[..., None]
    new_col1.fill(1)

    """Loading Hillary Data"""
    hill = pickle.load(open("h_tfidf_all.p", "rb"))

    hi = hill.toarray()
    print '\nHillary data shape'
    print hi.shape

    new_col2 = hi.sum(1)[..., None]
    new_col2.fill(0)

    """Combining the author's data"""
    new_col = np.append(new_col1, new_col2, 0)
    print '\nLabel size'
    print np.shape(new_col)

    temp_data = np.append(tr, hi, 0)
    print '\nCombined size:'
    print np.shape(temp_data)

    """Creating training data SVD arrays without labels"""
    print '\nFitting and transforming training data'
    pickle.dump(n, open("%s/num_dimen.p" % destination, "wb"))

    svd = TruncatedSVD(n_components=n)
    sv_data = svd.fit_transform(temp_data)  # Fit and transform of SVD to training data
    pickle.dump(svd, open("%s/svd_trained.p" % destination, "wb"))

    print '\nSVD size'
    print sv_data.shape
    print 'Training data variance with %d SVD components' % n
    var = svd.explained_variance_.sum() * 100
    print var
    with open('%s/%d_dimen_var.csv' % (destination, n), 'wb') as csvfile:
        w = csv.writer(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
        w.writerow(['Dimensions:%d, Variance:%d' % (n, var)])

    """Creating final training data SVD arrays with labels"""
    train_data = np.append(sv_data, new_col, 1)

    print '\nTraining data Size'
    print train_data.shape
    print '\nTrump ID:'
    print train_data[0][n]
    print '\nHillary ID:'
    print train_data[-1][n]

    pickle.dump(train_data, open("%s/labelled_svd_train.p" % destination, "wb"))
    # np.savetxt("labelled_2D_svd_train.csv", train_data, delimiter=',')


    """Creating test data SVD arrays without labels"""
    print '\nTransforming test data'
    test1 = pickle.load(open("oldtrump_tfidf_all.p", "rb"))
    test2 = pickle.load(open("oldhill_tfidf_all.p", "rb"))

    ts1 = svd.transform(test1)  # Transform of test data to SVD
    new_col1 = ts1.sum(1)[..., None]
    new_col1.fill(1)
    print 'Trump test data size'
    print ts1.shape

    ts2 = svd.transform(test2)
    new_col2 = ts2.sum(1)[..., None]
    new_col2.fill(0)
    print 'Hillary test data size'
    print ts2.shape

    """Creating final test data SVD arrays with labels"""
    print 'Number of SVD dimensions: %d' % n
    test_data1 = np.append(ts1, new_col1, 1)
    print '\nTest data 1 shape:'
    print test_data1.shape

    test_data2 = np.append(ts2, new_col2, 1)
    print '\nTest data 2 shape:'
    print test_data2.shape

    pickle.dump(test_data1, open("%s/oldT_lsa_labelled.p" % destination, "wb"))
    pickle.dump(test_data2, open("%s/oldH_lsa_labelled.p" % destination, "wb"))
