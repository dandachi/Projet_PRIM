# -*- coding: utf-8 -*-
"""
Created on Mon Feb 01 15:13:35 2016
spark-submit --driver-memory 2g spark-1.py ./input/ personalRatings.txt
@author: Dandachi
"""

#spark time
import sys
import itertools
import numpy as np
from math import sqrt
from operator import add
from os.path import join, isfile, dirname

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS

def get_error(Q, X, Y):
    error=0
    for r in Q:
        error=error+(r[2]-np.dot(X[r[0],:],Y[:,r[1]]))**2
    return error
    
def get_error_Qpred(Q, Qpred, W):
    return np.sum((W * (Q - Qpred))**2)
    
def get_error_Qpred_norm_un(Q, Qpred, W):
    return np.sum(np.abs(W * (Q - Qpred)))

def get_histogram(Q,Qpred,W):
    return np.floor(((W *(Q-Qpred)).ravel())[W.ravel()==1]).astype(int)
    #histogram entre -5 et 5

def mult_Y(Q,Y,u,m,k):
    QY=np.zeros([u,k])
    #print 'multy'
    for r in Q:
        QY[r[0],:]=QY[r[0],:]+r[2]*Y[r[1],:]
    return QY

def mult_X(Q,X,u,m,k):
    QX=np.zeros([m,k])
    #print 'multx'
    for r in Q:
        QX[r[1],:]=QX[r[1],:]+r[2]*X[r[0],:]
    return QX
    
def parseRating(line):
    """
    Parses a rating record in MovieLens format userId::movieId::rating::timestamp .
    """
    fields = line.strip().split("::")
    return long(fields[3]) % 10, (int(fields[0]), int(fields[1]), float(fields[2]))

def parseMovie(line):
    """
    Parses a movie record in MovieLens format movieId::movieTitle .
    """
    fields = line.strip().split("::")
    return int(fields[0]), fields[1]

def loadRatings(ratingsFile):
    """
    Load ratings from file.
    """
    if not isfile(ratingsFile):
        print "File %s does not exist." % ratingsFile
        sys.exit(1)
    f = open(ratingsFile, 'r')
    ratings = filter(lambda r: r[2] > 0, [parseRating(line)[1] for line in f])
    f.close()
    if not ratings:
        print "No ratings provided."
        sys.exit(1)
    else:
        return ratings

def computeRmse(model, data, n):
    """
    Compute RMSE (Root Mean Squared Error).
    """
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
      .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
      .values()
    return sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))

def myComputeRmse(modelX,modelY, data, n):
    """
    Compute RMSE (Root Mean Squared Error).
    error+
    """
    error=data.map(lambda x: (x[2]-np.dot(modelX[x[0],:],modelY[:,x[1]]))**2).reduce(add)
    return sqrt(error / float(n))

def ayvect(x,modelY):
    print 'movie'+str(x[1])
    return (x[0], x[2]*modelY[x[1],:])
    
def myALStrain(training, rank, numIter, lmbda,usersnum,moviesnum):
    np.random.seed(45)
    modelX = 5 * np.random.rand(usersnum+1, rank) 
    modelY = 5 * np.random.rand(moviesnum+1, rank)

    for ii in range(numIter):
        AY=np.zeros((usersnum+1,rank))
        AX=np.zeros((moviesnum+1,rank))
        AY_list=training.map(lambda x:(x[0], x[2]*modelY[x[1],:])).reduceByKey(lambda p,q:p+q).collectAsMap()
        
        for key, value in AY_list.iteritems():
            AY[key]=value
        modelX = np.linalg.solve(np.dot(modelY.T, modelY) + lmbda * np.eye(rank), 
                        AY.T).T
        
        AX_list=training.map(lambda x: (x[1], x[2]*modelX[x[0],:])).reduceByKey(lambda p,q:p+q).collectAsMap()
        
        for key, value in AX_list.iteritems():
            AX[key]=value
        modelY = np.linalg.solve(np.dot(modelX.T, modelX) + lmbda * np.eye(rank),
                        AX.T).T
        if ii % 10 == 0:
            print(str(ii)+'iteration  completed'.format(ii))
    return (modelX,modelY)

if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print "out: Usage: /path/to/spark/bin/spark-submit --driver-memory 2g " + \
          "MovieLensALS.py movieLensDataDir personalRatingsFile"
        sys.exit(1)

    # set up environment
    conf = SparkConf() \
      .setAppName("MovieLensALS") \
      .set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf)

    # load personal ratings
    myRatings = loadRatings(sys.argv[2])
    myRatingsRDD = sc.parallelize(myRatings, 1)
    
    # load ratings and movie titles

    movieLensHomeDir = sys.argv[1]

    # ratings is an RDD of (last digit of timestamp, (userId, movieId, rating))
    ratings = sc.textFile(join(movieLensHomeDir, "ratings.dat")).map(parseRating)

    # movies is an RDD of (movieId, movieTitle)
    movies = dict(sc.textFile(join(movieLensHomeDir, "movies.dat")).map(parseMovie).collect())

    numRatings = ratings.count()
    numUsers = ratings.values().map(lambda r: r[0]).distinct().count()
    numMovies = ratings.values().map(lambda r: r[1]).distinct().count()
    maxUsers=ratings.values().map(lambda r: r[0]).distinct().max()
    maxMovies=ratings.values().map(lambda r: r[1]).distinct().max()
    print "out: Got %d ratings from %d users on %d movies." % (numRatings, numUsers, numMovies)

    # split ratings into train (60%), validation (20%), and test (20%) based on the 
    # last digit of the timestamp, add myRatings to train, and cache them

    # training, validation, test are all RDDs of (userId, movieId, rating)

    numPartitions = 4
    training = ratings.filter(lambda x: x[0] < 6) \
      .values() \
      .union(myRatingsRDD) \
      .repartition(numPartitions) \
      .cache()

    validation = ratings.filter(lambda x: x[0] >= 6 and x[0] < 8) \
      .values() \
      .repartition(numPartitions) \
      .cache()

    test = ratings.filter(lambda x: x[0] >= 8).values().cache()

    numTraining = training.count()
    numValidation = validation.count()
    numTest = test.count()

    print "out: Training: %d, validation: %d, test: %d" % (numTraining, numValidation, numTest)

    # train models and evaluate them on the validation set

    ranks = [32, 64]
    lambdas = [0.1, 10.0]
    numIters = [10, 20]
    bestModel = None
    bestValidationRmse = float("inf")
    bestRank = 0
    bestLambda = -1.0
    bestNumIter = -1

    for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
        modelX,modelY = myALStrain(training, rank, numIter, lmbda,maxUsers,maxMovies)
        validationRmse = myComputeRmse(modelX,modelY, validation, numValidation)
        print "out: RMSE (validation) = %f for the model trained with " % validationRmse + \
              "rank = %d, lambda = %.1f, and numIter = %d." % (rank, lmbda, numIter)
        if (validationRmse < bestValidationRmse):
            bestModelX = modelX
            bestModelY=modelY
            bestValidationRmse = validationRmse
            bestRank = rank
            bestLambda = lmbda
            bestNumIter = numIter

    testRmse = myComputeRmse(bestModelX,bestModelY, test, numTest)
    # evaluate the best model on the test set
    print "out: The best model was trained with rank = %d and lambda = %.1f, " % (bestRank, bestLambda) \
      + "and numIter = %d, and its RMSE on the test set is %f." % (bestNumIter, testRmse)

    # compare the best model with a naive baseline that always returns the mean rating
    meanRating = training.union(validation).map(lambda x: x[2]).mean()
    baselineRmse = sqrt(test.map(lambda x: (meanRating - x[2]) ** 2).reduce(add) / numTest)
    improvement = (baselineRmse - testRmse) / baselineRmse * 100
    print "out: The best model improves the baseline by %.2f" % (improvement) + "%."

    # make personalized recommendations

#    myRatedMovieIds = set([x[1] for x in myRatings])
#    candidates = sc.parallelize([m for m in movies if m not in myRatedMovieIds])
#    predictions = bestModel.predictAll(candidates.map(lambda x: (0, x))).collect()
#    recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:50]
#
#    print "Movies recommended for you:"
#    for i in xrange(len(recommendations)):
#        print ("%2d: %s" % (i + 1, movies[recommendations[i][1]])).encode('ascii', 'ignore')

    # clean up
    sc.stop()    
    