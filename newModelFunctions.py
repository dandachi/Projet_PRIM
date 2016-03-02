import numpy as np 


def generate_pairs(label, n_pairs, positive_ratio, random_state=42):
    rng = np.random.RandomState(random_state)
    n_samples = label.shape[0]
    pairs_idx = np.zeros((n_pairs, 2), dtype=int)
    pairs_idx[:, 0] = rng.randint(0, n_samples, n_pairs)
    rand_vec = rng.rand(n_pairs)
    for i in range(n_pairs):
        if rand_vec[i] <= positive_ratio:
            idx_same = np.where(label == label[pairs_idx[i, 0]])[0]
            idx2 = rng.randint(idx_same.shape[0])
            pairs_idx[i, 1] = idx_same[idx2]
        else:
            idx_diff = np.where(label != label[pairs_idx[i, 0]])[0]
            idx2 = rng.randint(idx_diff.shape[0])
            pairs_idx[i, 1] = idx_diff[idx2]
    pairs_label = 2.0 * (label[pairs_idx[:, 0]] == label[pairs_idx[:, 1]]) - 1.0
    return pairs_idx, pairs_label

def genre_to_column(datamerged):
    
    datamerged['Action']=datamerged['genres'].apply(lambda x: int('Action' in x))
    datamerged['Adventure']=datamerged['genres'].apply(lambda x: int('Adventure' in x))
    datamerged['Animation']=datamerged['genres'].apply(lambda x: int('Animation' in x))
    datamerged['Children']=datamerged['genres'].apply(lambda x: int('Children\'s' in x))
    datamerged['Comedy']=datamerged['genres'].apply(lambda x: int('Comedy' in x))
    datamerged['Crime']=datamerged['genres'].apply(lambda x: int('Crime' in x))
    datamerged['Documentary']=datamerged['genres'].apply(lambda x: int('Documentary' in x))
    datamerged['Drama']=datamerged['genres'].apply(lambda x: int('Drama' in x))
    datamerged['Fantasy']=datamerged['genres'].apply(lambda x: int('Fantasy' in x))
    datamerged['Fnoir']=datamerged['genres'].apply(lambda x: int('Film-Noir' in x))
    datamerged['Horror']=datamerged['genres'].apply(lambda x: int('Horror' in x))
    datamerged['Musical']=datamerged['genres'].apply(lambda x: int('Musical' in x))
    datamerged['Mystery']=datamerged['genres'].apply(lambda x: int('Mystery' in x))
    datamerged['Romance']=datamerged['genres'].apply(lambda x: int('Romance' in x))
    datamerged['SciFi']=datamerged['genres'].apply(lambda x: int('Sci-Fi' in x))
    datamerged['Thriller']=datamerged['genres'].apply(lambda x: int('Thriller' in x))
    datamerged['War']=datamerged['genres'].apply(lambda x: int('War' in x))
    datamerged['Western']=datamerged['genres'].apply(lambda x: int('Western' in x))
    return datamerged

def normalizeAll(datamerged):
    datamerged['Action']=datamerged['Action']/np.linalg.norm(datamerged['Action'])
    datamerged['Adventure']=datamerged['Adventure']/np.linalg.norm(datamerged['Adventure'])
    datamerged['Animation']=datamerged['Animation']/np.linalg.norm(datamerged['Animation'])
    datamerged['Children']=datamerged['Children']/np.linalg.norm(datamerged['Children'])
    datamerged['Comedy']=datamerged['Comedy']/np.linalg.norm(datamerged['Comedy'])
    datamerged['Crime']=datamerged['Crime']/np.linalg.norm(datamerged['Crime'])
    datamerged['Documentary']=datamerged['Documentary']/np.linalg.norm(datamerged['Documentary'])
    datamerged['Drama']=datamerged['Drama']/np.linalg.norm(datamerged['Drama'])
    datamerged['Fantasy']=datamerged['Fantasy']/np.linalg.norm(datamerged['Fantasy'])
    datamerged['Fnoir']=datamerged['Fnoir']/np.linalg.norm(datamerged['Fnoir'])
    datamerged['Horror']=datamerged['Horror']/np.linalg.norm(datamerged['Horror'])
    datamerged['Musical']=datamerged['Musical']/np.linalg.norm(datamerged['Musical'])
    datamerged['Romance']=datamerged['Romance']/np.linalg.norm(datamerged['Romance'])
    datamerged['SciFi']=datamerged['SciFi']/np.linalg.norm(datamerged['SciFi'])
    datamerged['Thriller']=datamerged['Thriller']/np.linalg.norm(datamerged['Thriller'])
    datamerged['War']=datamerged['War']/np.linalg.norm(datamerged['War'])
    datamerged['Western']=datamerged['Western']/np.linalg.norm(datamerged['Western'])
    datamerged['movie_id']=datamerged['movie_id']/np.linalg.norm(datamerged['movie_id'])
    datamerged['user_id']=datamerged['user_id']/np.linalg.norm(datamerged['user_id'])
    datamerged['zip-code']=datamerged['zip-code']/np.linalg.norm(datamerged['zip-code'])
    datamerged['genderbinary']=datamerged['genderbinary']/np.linalg.norm(datamerged['genderbinary'])
    return datamerged
    
def euc_dist_pairs(X, pairs, batch_size=10000):
    """Compute an array of Euclidean distances between points indexed by pairs

    To make it memory-efficient, we compute the array in several batches.
    
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Data matrix
    pairs : array, shape (n_pairs, 2)
        Pair indices
    batch_size : int
        Batch size (the smaller, the slower but less memory intensive)
        
    Output
    ------
    dist : array, shape (n_pairs,)
        The array of distances
    """
    n_pairs = pairs.shape[0]
    dist = np.ones((n_pairs,), dtype=np.dtype("float32"))
    for a in range(0, n_pairs, batch_size):
        b = min(a + batch_size, n_pairs)
        dist[a:b] = np.sqrt(np.sum((X[pairs[a:b, 0], :] - X[pairs[a:b, 1], :]) ** 2, axis=1))
    return dist

def mal_dist_pairs(X, pairs,M, batch_size=10000):
    """Compute an array of Euclidean distances between points indexed by pairs

    To make it memory-efficient, we compute the array in several batches.
    
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Data matrix
    pairs : array, shape (n_pairs, 2)
        Pair indices
    batch_size : int
        Batch size (the smaller, the slower but less memory intensive)
        
    Output
    ------
    dist : array, shape (n_pairs,)
        The array of distances
    """
    n_pairs = pairs.shape[0]
    dist = np.ones((n_pairs,), dtype=np.dtype("float32"))
    
    for a in range(0, n_pairs, batch_size):
        b = min(a + batch_size, n_pairs)
        diff = X[pairs[a:b, 0], :] - X[pairs[a:b, 1], :]
        dist[a:b] =np.sum(np.dot(M, diff.T) * diff.T, axis=0)
    return dist

def psd_proj(M):
    """ projection de la matrice M sur le cone des matrices semi-definies
    positives"""
    # calcule des valeurs et vecteurs propres
    eigenval, eigenvec = np.linalg.eigh(M)
    # on trouve les valeurs propres negatives ou tres proches de 0
    ind_pos = eigenval > 1e-10
    # on reconstruit la matrice en ignorant ces dernieres
    M = np.dot(eigenvec[:, ind_pos] * eigenval[ind_pos][np.newaxis, :],
               eigenvec[:, ind_pos].T)
    return M


def hinge_loss_pairs(X, pairs_idx, y_pairs, M,b):
    """Calcul du hinge loss sur les paires
    """
    diff = X[pairs_idx[:, 0], :] - X[pairs_idx[:, 1], :]
    return np.maximum(0.0, 1 + y_pairs.T * (np.sum(
                                 np.dot(M, diff.T) * diff.T, axis=0) - 2))
    #pobj[t]=max(0,1-np.dot(np.dot(X[idx],w),y[idx]))
    #pobj[t]=max(0,1-np.dot(np.dot(X[idx],w),y[idx]))
    #gradient = -X[idx, :] * y[idx]                             

def mse_loss_pairs(X,pairs_idx,y_pairs,M):
    diff = X[pairs_idx[:, 0], :] - X[pairs_idx[:, 1], :]    
    return 0.5*np.mean((y_pairs.T-(np.sum(np.dot(M, diff.T) * diff.T, axis=0)-2))**2)
    #pobj[t] = 0.5 * np.mean((y - np.dot(X, w)) ** 2)
    #gradient = X[idx, :] * (np.dot(X[idx], w) - y[idx])+alpha*np.sum(w)
    
        
                         
def sgd_metric_learning(X,y,pairs_idx,pairs_label, gamma,alpha, n_iter, n_eval, M_ini, random_state=42, batch_size=10,b=2):
    """Stochastic gradient algorithm for metric learning
    
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,)
        The targets.
    gamma : float | callable
        The step size. Can be a constant float or a function
        that allows to have a variable step size
    n_iter : int
        The number of iterations
    n_eval : int
        The number of pairs to evaluate the objective function
    M_ini : array, shape (n_features,n_features)
        The initial value of M
    random_state : int
        Random seed to make the algorithm deterministic
    """
    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    # tirer n_eval paires aleatoirement
    #pairs_idx = rng.randint(0, n_samples, (n_eval, 2))
    # calcul du label des paires
    #y_pairs = 2.0 * (y[pairs_idx[:, 0]] == y[pairs_idx[:, 1]]) - 1.0
    y_pairs=pairs_label
    M = M_ini.copy()
    pobj = np.zeros(n_iter)
    _,unique_y_idx=np.unique(y,return_index=True)

    if not callable(gamma):
        # Turn gamma to a function for QUESTION 5
        gamma_func = lambda t: gamma
    else:
        gamma_func = gamma
    
    if not callable(alpha):
        # Turn gamma to a function for QUESTION 5
        alpha_func = lambda t: alpha
    else:
        alpha_func = alpha

    for t in range(n_iter):
       # print "iteration :",t
        pobj[t] = np.mean(hinge_loss_pairs(X, pairs_idx, y_pairs, M,0.5))+(alpha_func(t)/2.0)*np.linalg.norm(M)
#        pobj[t]= np.mean(mse_loss_pairs(X,pairs_idx,y_pairs,M))+(alpha/2.0)*np.linalg.norm(M)
        #print 'loss ',pobj[t]
        idx = rng.randint(0, n_samples, 2)
        
        diff = X[idx[0], :] - X[idx[1], :]
        y_idx = 2.0 * (y[idx[0]] == y[idx[1]]) - 1.0
        #print 'y',y_idx
        #print 'diff',np.dot(diff, np.dot(M, diff.T))
        #print y_idx
        #print((1.0 + y_idx * (np.dot(diff, np.dot(M, diff.T))/100.0 - 2.0)))
        hinge_factor=(1 + y_idx * (np.dot(diff, np.dot(M, diff.T))-0.5)) > 0
        gradient = y_idx * np.outer(diff, diff) *(hinge_factor)+alpha_func(t)*M
        #print gradient
#        gradient = -np.outer(diff, diff)*np.mean(y_idx.T-(np.sum(np.dot(M, diff.T) * diff.T, axis=0)-2))+ alpha*M  
        M -= gamma_func(t) * gradient
        M = psd_proj(M)
    return M, pobj
    
    
def predict_M(Xtrain,Ytrain,Xtest,M, random_state=42):

    Ytest=np.zeros(Xtest.shape[0])
    for i in range(0,Xtest.shape[0]):
        #print "i#:",i  
        xdiff=np.where(Xtrain[:,0]==Xtest[i,0])[0]
        diff = Xtrain[xdiff]-Xtest[i]
        distpredi=np.sum(np.dot(M, diff.T) * diff.T, axis=0)
        Ytest[i]=Ytrain[xdiff[np.argmin(distpredi)]]
    return Ytest
    
def mygamma(t):
    return (0.00001/np.log(t+1))
    
def myalpha(t):
    return 0.00001*(1.0/(t+1))    