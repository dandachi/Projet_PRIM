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

from pyspark import SparkConf, SparkContext, storagelevel
from pyspark.mllib.random import RandomRDDs
from pyspark.mllib.recommendation import ALS
import time
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
    error=data.map(lambda x: (x[2]-np.dot(modelX[x[0],:],modelY[x[1],:]))**2).reduce(add)
    return sqrt(error / float(n))

def ayvect(x,modelY):
    print 'movie'+str(x[1])
    return (x[0], x[2]*modelY[x[1],:])
    
def sparkSolveX(iterai,Y,m,k,lambda_):
    tempai=np.zeros(m+1)

    for atuple in iterai:
        tempai[atuple[0]]=atuple[1]
    
    tempwi = tempai>0.5
    tempwi[tempwi == True] = 1.0
    tempwi[tempwi == False] = 0.0
    x=np.linalg.solve(np.dot(Y.T, np.array([tempwi,]*k).T*Y) + lambda_ * np.eye(k), 
                    np.dot(Y.T, tempai.T)).T
    return x

def sparkSolveY(iteraj,X,n,k,lambda_):
    tempaj=np.zeros(n+1)

    for atuple in iteraj:
        tempaj[atuple[0]]=atuple[1]
    
    tempwj = tempaj>0.5
    tempwj[tempwj == True] = 1.0
    tempwj[tempwj == False] = 0.0
    
     
    y = np.linalg.solve(np.dot(X.T,np.array([tempwj,]*k).T* X) + lambda_ * np.eye(k),
                        np.dot(X.T, tempaj.T))
                        
    return y
    
def myALStrain(training, rank, numIter, lmbda,usersnum,moviesnum,ratings):
    np.random.seed(45)
    modelX = 5 * np.random.rand(usersnum+1, rank) 
    modelY = 5 * np.random.rand(moviesnum+1, rank)
    #modelX=RandomRDDs.normalVectorRDD(sc, usersnum+1, rank,4, seed=45)
    #modelY=RandomRDDs.normalVectorRDD(sc, moviesnum+1, rank,4, seed=45)
    ai=training.map(lambda x: (x[0],(x[1],x[2]))).groupByKey()
    aj=training.map(lambda x: (x[1],(x[0],x[2]))).groupByKey()
    for ii in range(numIter):
     
        
        mxByligne = ai.map(lambda x: (x[0],sparkSolveX(x[1],modelY,moviesnum,rank,lmbda))).collectAsMap()
        for key, value in mxByligne.iteritems():
            modelX[key]=value
        
        myByligne = aj.map(lambda x: (x[0],sparkSolveY(x[1],modelX,usersnum,rank,lmbda))).collectAsMap()
        for key, value in myByligne.iteritems():
            modelY[key]=value
                        

        
        if ii % 10 == 0:
            print(str(ii)+'iteration  completed'.format(ii))
    return modelX,modelY

if __name__ == "__main__":

    if (len(sys.argv) != 3):
        print "out: Usage: /path/to/spark/bin/spark-submit --driver-memory 2g " + \
          "MovieLensALS.py movieLensDataDir personalRatingsFile"
        sys.exit(1)

    # set up environment
    conf = SparkConf() \
      .setAppName("MovieLensALS") \
      #.set("spark.executor.memory", "4g")
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
    training = ratings.filter(lambda x: x[0] < 8) \
      .values() \
      .union(myRatingsRDD) \
      .repartition(numPartitions) \
      .cache()
      #.persist(storagelevel.StorageLevel.useDisk)

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
    start = time.time()
    ranks = [4]
    lambdas = [0.1]
    numIters = [20]
    bestModel = None
    bestValidationRmse = float("inf")
    bestRank = 0
    bestLambda = -1.0
    bestNumIter = -1

    for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
        modelX,modelY = myALStrain(training, rank, numIter, lmbda,maxUsers,maxMovies,ratings)
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
    
    end = time.time()
    
    print 'total time = ' + str(end-start)
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
    print 'total time = ' + str(end-start)
    # clean up
    sc.stop()    
    