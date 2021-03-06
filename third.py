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


tag_headers = ['user_id', 'movie_id', 'tag', 'timestamp']
tags = pd.read_table('users.dat', sep='::', header=None, names=tag_headers)

rating_headers = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('ratings.dat', sep='::', header=None, names=rating_headers)

movie_headers = ['movie_id', 'title', 'genres']
movies = pd.read_table('movies.dat',
                       sep='::', header=None, names=movie_headers)
movie_titles = movies.title.tolist()

df = ratings.join(movies, on=['movie_id'], rsuffix='_r').join(tags, on=['movie_id'], rsuffix='_t')
del df['movie_id_r']
del df['user_id_t']
del df['movie_id_t']
del df['timestamp_t']

####################### Preparing training data and testing data
random.seed(1992)
#df_train=df.sample(frac=0.9).sort_index()
#df_test = df.loc[~df.index.isin(df_train.index)]
df_train=df.loc[df['timestamp']%10<8]
df_test=df.loc[df['timestamp']%10>=8]

#rp = df.pivot_table(columns=['movie_id'],index=['user_id'],values='rating')
##rp_train.columns.reindex(range(1,len(movie_titles)))
#rp = rp.fillna(0); # Replace NaN
#rp=  rp.reindex(range(1,len(tags)+1),range(1,len(movie_titles)+1),fill_value=0)
#Q = rp.values

rp_train = df_train.pivot_table(columns=['movie_id'],index=['user_id'],values='rating')
#rp_train.columns.reindex(range(1,len(movie_titles)))
rp_train = rp_train.fillna(0); # Replace NaN
rp_train=  rp_train.reindex(range(1,len(tags)+1),range(1,len(movie_titles)+1),fill_value=0)
Q_train = rp_train.values

rp_test = df_test.pivot_table(columns=['movie_id'],index=['user_id'],values='rating')
#rp_train.columns.reindex(range(1,len(movie_titles)))
rp_test = rp_test.fillna(0); # Replace NaN
rp_test=  rp_test.reindex(range(1,len(tags)+1),range(1,len(movie_titles)+1),fill_value=0)
Q_test = rp_test.values


#W = Q>0.5
#W[W == True] = 1
#W[W == False] = 0
## To be consistent with our Q matrix
#W = W.astype(np.float64, copy=False)

W_train = Q_train>0.5
W_train[W_train == True] = 1
W_train[W_train == False] = 0
# To be consistent with our Q matrix
W_train = W_train.astype(np.float64, copy=False)

W_test = Q_test>0.5
W_test[W_test == True] = 1
W_test[W_test == False] = 0
# To be consistent with our Q matrix
W_test = W_test.astype(np.float64, copy=False)

W=W_train+W_test
Q=Q_train+Q_test

start=time.time()
lambda_ = 0.1
#n_factors = [2,4,8,16,32,64,128,256,512,1024,2048]
#n_factors=range(1,48,2)
n_factors=[32]
n, m = Q_train.shape
n_iterations = 20
k_errors=np.zeros(len(n_factors))
gradient=0
pas=1
Q_hats=[]
for k_factor in n_factors:
    np.random.seed(45)
    X = 5 * np.random.rand(n, k_factor) 
    Y = 5 * np.random.rand(k_factor, m)
    errors = []
    for ii in range(n_iterations):
        X = np.linalg.solve(np.dot(Y, Y.T) + lambda_ * np.eye(k_factor), 
                        np.dot(Y, Q_train.T)).T
        Y = np.linalg.solve(np.dot(X.T, X) + lambda_ * np.eye(k_factor),
                        np.dot(X.T, Q_train))
        if ii % 100 == 0:
            print('{}th iteration is completed'.format(ii))
        #errors.append(get_error(Q_test, X, Y, W_test))
    Q_hat = np.dot(X, Y)#.clip(min=0,max=5)
    #Q_hats.append(Q_hat)
    k_errors[n_factors.index(k_factor)]=np.sqrt(get_error_Qpred(Q_test,Q_hat,W_test)/np.sum(W_test))

end=time.time()
print 'execution time',(end-start)

Q_baseline=np.ones((n,m))*(np.sum(Q_train)/np.sum(W_train))
rmse_baseline=np.sqrt(get_error_Qpred(Q_test,Q_baseline,W_test)/np.sum(W_test))
print 'rmse baseline= ',rmse_baseline

plt.plfig = plt.figure(2029)
plt.xlabel('rank')
plt.ylabel('RMSE')
plt.title('rmse to rank variation')
plt.plot(n_factors,k_errors)
plt.savefig('rmse_rank_model1_zoomed.svg')

Allhists=np.zeros((len(n_factors),int(np.sum(W_test))))
for h in range(0,len(n_factors)):
    Allhists[h]=get_histogram(Q_test,Q_hats[h],W_test)

for h in range(0,len(n_factors)):
    fig=plt.figure(2000+h)
    plt.xlabel(n_factors[h])
    plt.hist(Allhists[h],bins=[-5,-4,-3,-2,-1,0,1,2,3,4,5])
    plt.savefig('hist'+str(n_factors[h])+'.png')
    
#moyen
moyAll=np.mean(Allhists,axis=1)
stdAll=np.std(Allhists,axis=1)  
fig=plt.figure(3001)
plt.plot(n_factors,moyAll,n_factors,stdAll)  

    