import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import math


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
    
os.chdir("C:\Users\Dandachi\Desktop\Projet_PRIM\ml-latest")


tag_headers = ['user_id', 'movie_id', 'tag', 'timestamp']
tags = pd.read_table('tags.csv', sep=',', header=None, names=tag_headers)

rating_headers = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('ratings.csv', sep=',', header=None, names=rating_headers)

movie_headers = ['movie_id', 'title', 'genres']
movies = pd.read_table('movies.csv',
                       sep=',', header=None, names=movie_headers)
movie_titles = movies.title.tolist()

df = ratings.join(movies, on=['movie_id'], rsuffix='_r').join(tags, on=['movie_id'], rsuffix='_t')
del df['movie_id_r']
del df['user_id_t']
del df['movie_id_t']
del df['timestamp_t']

####################### Preparing training data and testing data
m=tags.shape[0]
n=movies.shape[0]
random.seed(1992)
df=df[df.movie_id < n]
df=df[df.user_id<m]
df_train=df.sample(frac=0.9).sort_index()
Q_train=df_train[['user_id','movie_id','rating']].values
df_test = df.loc[~df.index.isin(df_train.index)]
Q_test=df_test[['user_id','movie_id','rating']].values


lambda_ = 0.1
n_factors = [2,4,8,16,32,64,128,256,512,1024,2048]
#n_factors=range(1,65)
#n_factors=[30]

n_iterations = 20
k_errors=np.zeros(len(n_factors))
gradient=0
pas=1
Q_hats=[]
for k_factor in n_factors:
    np.random.seed(45)
    X = 5 * np.random.rand(m, k_factor) 
    Y = 5 * np.random.rand(k_factor, n)
    errors = []
    for ii in range(n_iterations):
        AY=mult_Y(Q_train,Y.T,m,n,k_factor)
        X = np.linalg.solve(np.dot(Y, Y.T) + lambda_ * np.eye(k_factor), 
                        AY.T).T
        AX=mult_X(Q_train,X,m,n,k_factor)
        Y = np.linalg.solve(np.dot(X.T, X) + lambda_ * np.eye(k_factor),
                        AX.T)
        if ii % 100 == 0:
            print('{}th iteration is completed'.format(ii))
        #errors.append(get_error(Q_test, X, Y, W_test))
    k_errors[n_factors.index(k_factor)]=get_error(Q_test,X,Y)
#plt.plot(errors);
#plt.ylim([0, 1000000]);

plt.plot(n_factors,np.sqrt(k_errors/Q_train.shape[0]))
plt.ylim([650000, 1400000]);

Allhists=np.zeros((len(n_factors),int(np.sum(W_test))))
for h in range(0,len(n_factors)):
    Allhists[h]=get_histogram(Q_test,Q_hats[h],W_test)

for h in range(0,len(n_factors)):
    fig=plt.figure(2000+h)
    plt.xlabel(n_factors[h])
    plt.hist(Allhists[h],bins=[-5,-4,-3,-2,-1,0,1,2,3,4,5])
    plt.savefig('histsdqsd'+str(n_factors[h])+'.svg')
    
#moyen
moyAll=np.mean(Allhists,axis=1)
stdAll=np.std(Allhists,axis=1)  
fig=plt.figure(3001)
plt.plot(n_factors,moyAll,n_factors,stdAll)  

    