import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import math
import time


def get_error(Q, X, Y, W):
    return np.sum((W * (Q - np.dot(X, Y)))**2)
    
def get_error_Qpred(Q, Qpred, W):
    return np.sum((W * (Q - Qpred))**2)
    
def get_error_Qpred_norm_un(Q, Qpred, W):
    return np.sum(np.abs(W * (Q - Qpred)))

def get_histogram(Q,Qpred,W):
    return np.floor(((W *(Q-Qpred)).ravel())[W.ravel()==1]).astype(int)
    #histogram entre -5 et 5
    
os.chdir("C:\Users\Dandachi\Desktop\Projet_PRIM")

users_headers = ['user_id','gender','age', 'occupation', 'zip-code']
users = pd.read_table('users.dat', sep='::', header=None, names=users_headers)

rating_headers = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('ratings.dat', sep='::', header=None, names=rating_headers)

movie_headers = ['movie_id', 'title', 'genres']
movies = pd.read_table('movies.dat',
                       sep='::', header=None, names=movie_headers)
movie_titles = movies.title.tolist()

datamerged = (ratings.merge(movies, on=['movie_id'])).merge(users, on=['user_id'])
data_merged=datamerged.sort_values('user_id')

####################### Preparing training data and testing data
random.seed(1992)
#df_train=df.sample(frac=0.9).sort_index()
#df_test = df.loc[~df.index.isin(df_train.index)]
df_train=datamerged.loc[datamerged['timestamp']%10<8]
df_test=datamerged.loc[datamerged['timestamp']%10>=8]


rp_train = df_train.pivot_table(columns=['movie_id'],index=['user_id'],values='rating')
rp_train = rp_train.fillna(0); # Replace NaN
rp_train=  rp_train.reindex(range(1,len(users)+1),range(1,len(movie_titles)+1),fill_value=0)
Q_train = rp_train.values

rp_test = df_test.pivot_table(columns=['movie_id'],index=['user_id'],values='rating')
#rp_train.columns.reindex(range(1,len(movie_titles)))
rp_test = rp_test.fillna(0); # Replace NaN
rp_test=  rp_test.reindex(range(1,len(users)+1),range(1,len(movie_titles)+1),fill_value=0)
Q_test = rp_test.values



W_train = Q_train>0.5
W_train[W_train == True] = 1
W_train[W_train == False] = 0
W_train = W_train.astype(np.float64, copy=False)

W_test = Q_test>0.5
W_test[W_test == True] = 1
W_test[W_test == False] = 0
W_test = W_test.astype(np.float64, copy=False)

W=W_train+W_test
Q=Q_train+Q_test

start=time.time()
lambda_ = 0.1
#n_factors = [2,4,8,16,32,64,128,256,512,1024,2048]
#n_factors=range(1,24)
n_factors=[4]
n, m = Q_train.shape
n_iterations = 20
k_errors=np.zeros(len(n_factors))
gradient=0
pas=1
Q_hats=[]
for k_factor in n_factors:
    np.random.seed(45)
    X = 5 * np.random.rand(n, k_factor) 
    Y = 5 * np.random.rand(m,k_factor)
    wspread=np.ones(k_factor)
    #errors = []
    for ii in range(n_iterations):
        for u, Wu in enumerate(W_train):
            X[u] = np.linalg.solve(np.dot(Y.T, np.array([Wu,]*k_factor).T*Y) + lambda_ * np.eye(k_factor), 
                    np.dot(Y.T, Q_train[u].T)).T
        for i, Wi in enumerate(W_train.T):
            Y[i] = np.linalg.solve(np.dot(X.T,np.array([Wi,]*k_factor).T* X) + lambda_ * np.eye(k_factor),
                        np.dot(X.T, Q_train[:, i]))
        
        print('{}th iteration is completed'.format(ii))
        #errors.append(np.sqrt(get_error_Qpred(Q_test,np.dot(X, Y.T),W_test)/np.sum(W_test)))
    Q_hat = np.dot(X, Y.T)#.clip(min=0,max=5)
    Q_hats.append(Q_hat)
    k_errors[n_factors.index(k_factor)]=np.sqrt(get_error_Qpred(Q_test,Q_hat,W_test)/np.sum(W_test))
end=time.time()
print 'total time = ' + str(end-start)
print k_errors
plt.plot(n_factors,k_errors);
plt.ylim([3, 3.2]);

plt.plot(n_factors,k_errors)
plt.ylim([650000, 1400000]);

Allhists=np.zeros((len(n_factors),int(np.sum(W_test))))
for h in range(0,len(n_factors)):
    Allhists[h]=get_histogram(Q_test,Q_hat,W_test)

for h in range(0,len(n_factors)):
    fig=plt.figure(2000+h)
    plt.xlabel('rating difference')
    plt.ylabel('count')
    plt.hist(Allhists[h],bins=[-5,-4,-3,-2,-1,0,1,2,3,4,5])
    plt.savefig('hist'+str(n_factors[h])+'.svg')
    
#moyen
moyAll=np.mean(Allhists,axis=1)
stdAll=np.std(Allhists,axis=1)  
fig=plt.figure(3001)
#n= 24 , rmse = 1.12530387
#errors=[1.1213583849485158,
# 1.0949271753962766,
# 1.0897329060749132,
# 1.0888709225900921,
# 1.0917134722328095,
# 1.0943583004908919,
# 1.097088270779226,
# 1.099269918472723,
# 1.1014207300311882,
# 1.1037473681291181,
# 1.1064507009371587,
# 1.1092125818050831,
# 1.1118180058111184,
# 1.1140623702074486,
# 1.1161024801577724,
# 1.118028495133127,
# 1.1198617017270847,
# 1.1216866058337229,
# 1.1235286232449078,
# 1.1253038711419006]
#n= 16 , rmse = 0.99123483
#n= 12 , rmse = 0.94044448
#n=  8 , rmse = 0.90080001
#n=  6 , rmse = 0.88332996
#n=  5 , rmse = 0.87809051 
#errors= [0.95286914470526629,
# 0.93063502136234699,
# 0.91263902033579192,
# 0.90101593281100012,
# 0.89230707143509114,
# 0.88658820435674213,
# 0.88285906498570321,
# 0.8807516167408419,
# 0.87965111755866532,
# 0.87894197981215583,
# 0.87846615314329246,
# 0.87813907471822905,
# 0.87793937544583955,
# 0.8778542549053695,
# 0.877844309708254,
# 0.87786756991317516,
# 0.87791169450839779,
# 0.8779697892450441,
# 0.87803219927386911,
# 0.87809051259996385]
# n=  5,
#[0.95248800986924187, 0.90646191581233682, 0.8912780210279676, 0.88381579678283384, 0.88094762020384743, 0.87936849608229617, 0.87833271166097104, 0.87765915207073197, 0.87725417255839866, 0.87700727932634015, 0.8768551676746652, 0.87676701976269678, 0.8767238825724879, 0.87670912736494699, 0.87671085438028173]
# ALS = RMSE (validation) = 0.864853 for the model trained with rank = 24, lambda = 0.1,
#kerros[ 0.91326811  0.88555236  0.87824889  0.87683382  0.87809051  0.88332996
#  0.89002202  0.90080001  0.91010274  0.92778239  0.94222026  0.95483231
#  0.97278485  0.98927319  1.0075013   1.01820604  1.03500993  1.04214392
#  1.06047551  1.07530225  1.09364333  1.10109389  1.11327455]

fig= plt.figure(2134)
plt.xlabel('iterations')
plt.ylabel('RMSE')
plt.title('rmse rank=24 equation with mask')
plt.plot(errors)
plt.savefig('rmse rank=24 equation with mask.svg')

fig = plt.figure(2202)
plt.xlabel('rank')
plt.ylabel('RMSE')
plt.title('rmse to rank variation')
plt.plot(n_factors,k_errors)
plt.savefig('rmse to rank variation.svg')