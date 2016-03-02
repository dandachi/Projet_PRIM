from newModelFunctions import *
import matplotlib.pyplot as plt
import os
import random
import pylab
import pandas as pd
import time
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
datamerged=datamerged.sort_values('user_id')
del datamerged['title']
datamerged=genre_to_column(datamerged)
del datamerged['genres']
datamerged['genderbinary']=np.where(datamerged['gender']=='M',1L,0L)
datamerged['zip-code']=datamerged['zip-code'].apply(lambda x: int(x[:5]))
del datamerged['gender']

datamerged['movie_id']=datamerged['movie_id']/np.linalg.norm(datamerged['movie_id'])
datamerged['user_id']=datamerged['user_id']/np.linalg.norm(datamerged['user_id'])
datamerged['zip-code']=datamerged['zip-code']/np.linalg.norm(datamerged['zip-code'])
#datamerged=normalizeAll(datamerged)
####################### Preparing training data and testing data
random.seed(1992)
#df_train=df.sample(frac=0.9).sort_index()
#df_test = df.loc[~df.index.isin(df_train.index)]
df_train=datamerged.loc[datamerged['timestamp']%10<8]
df_test=datamerged.loc[datamerged['timestamp']%10>=8]
Y_train=np.array(df_train['rating']).astype(np.float64)
Y_test=np.array(df_test['rating']).astype(np.float64)
del df_train['timestamp']
del df_train['rating']
del df_test['timestamp']
del df_test['rating']
#del df_train['user_id']
#del df_test['user_id']
#del df_train['zip-code']
#del df_test['zip-code']

X_train=np.array(df_train).astype(np.float64)
X_test=np.array(df_test).astype(np.float64)



#########Mahallanobis#########################################################
pairs_idx, pairs_label = generate_pairs(Y_train, 1000, 0.1)
def mygamma(t):
    return 0.001/np.sqrt(t+1)

start=time.time()
M_init=np.random.rand(X_train.shape[1],X_train.shape[1])
M,pobj=sgd_metric_learning(X_train,Y_train,pairs_idx,pairs_label,mygamma,0.00000001, 100000, 1000, M_init)
plt.plot(range(pobj.size),np.log(pobj))
dist = mal_dist_pairs(X_train, pairs_idx,M)
YprediMal=predict_M(X_train,Y_train,X_test,M)
errorPrediMal=np.sqrt(np.sum((YprediMal-Y_test)**2)/Y_test.shape[0])
end=time.time()
print 'total time=',(end-start)
#errorpredimal=1.36282479457
wut=np.sum(np.dot(M,X_train.T)*X_train.T,axis=0)
wut
#plt.plot(np.sort(wut))
fig= plt.figure(5119)
plt.xlabel('distance X.M.X\' ')
plt.ylabel('Index of Xtrain')
plt.title('RMSE error for Random forest prediction Vs. basic mean estimation')
plt.scatter(wut[:100],range(0,100),c=Y_train[:100],cmap=pylab.cm.coolwarm)
plt.savefig()
###############################################################################




## forest classifier###########################################################
from sklearn.ensemble import RandomForestClassifier


estimators=range(1,30)
errorForest=np.zeros(len(estimators))
for n_estimator in estimators:
    
    rf = RandomForestClassifier(n_estimators=n_estimator,n_jobs=7)
    rf.fit(X_train, Y_train)
    Y_predi=rf.predict(X_test)
    errorForest[estimators.index(n_estimator)]=np.sqrt(np.sum((Y_predi-Y_test)**2)/Y_test.shape[0])
#estimators=range(1,30)
#[ 1.36414204  1.4097892   1.40623183  1.34498764  1.32008758  1.31845391
#  1.30945917  1.30184244  1.3013619   1.29671717  1.2936509   1.29209268
#  1.29150348  1.28944406  1.28596007  1.28571393  1.28461155  1.28535048
#  1.28433189  1.28505535  1.28377045  1.28552049  1.28257435  1.28292292
#  1.28299145  1.2831559   1.28342798  1.28466043  1.28277019]

Y_moy=np.ones(Y_test.shape[0])*np.mean(Y_train)    
error_moy=np.ones(len(estimators))*np.sqrt(np.sum((Y_moy-Y_test)**2)/Y_test.shape[0])




fig= plt.figure(2134)
plt.xlabel('n_estimator for RF')
plt.ylabel('RMSE')
plt.title('RMSE error for Random forest prediction Vs. basic mean estimation')
plt.plot(estimators,errorForest,'r',estimators,error_moy)
plt.savefig('estimatorsForestvsMoy.svg')
###############################################################################



######Kmeans Classifier########################################################
from sklearn.cluster import MiniBatchKMeans

ncluster=1024
kmc = MiniBatchKMeans(n_clusters=ncluster)
kmc.fit(X_train, Y_train)
Ytrainkmc=kmc.predict(X_train)
Ytestkmc=kmc.predict(X_test)
Ypredikmc=np.zeros(X_test.shape[0])
Ypredikmc_corr=np.zeros(X_test.shape[0])
classToRating=np.zeros(ncluster)
for i in range(0,ncluster):
    classToRating[i]=np.mean(Y_train[np.where(Ytrainkmc==i)])

mctr=np.nanmin(classToRating)
Maxctr=np.nanmax(classToRating)
ctrRe=(classToRating-mctr)*4/(Maxctr-mctr)
ctrRe=np.rint(ctrRe+1)
Ypredikmc_corr[:]=ctrRe[Ytestkmc[:]]
error_kmc_corrected=np.sqrt(np.sum((Ypredikmc_corr-Y_test)**2)/Y_test.shape[0])
#ncluster=1000 error_kmc_corrected=1.2359261004867912

Ypredikmc[:]=np.rint(classToRating[Ytestkmc[:]])
    
error_kmc_moy=np.sqrt(np.sum((Ypredikmc-Y_test)**2)/Y_test.shape[0])
#ncluster=422  error_kmc_moy=1.1532711360122871
#ncluster=2000 error_kmc_moy=1.133738019039054

##############################################################################

