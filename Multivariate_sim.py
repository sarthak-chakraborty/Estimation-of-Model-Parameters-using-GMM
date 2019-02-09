
# coding: utf-8

# In[ ]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
from numpy import array
import numpy as np
import math
from scipy.stats import norm
import scipy.stats as stats
from sklearn import mixture
import sklearn
from sklearn.neighbors.kde import KernelDensity
from scipy.stats import gaussian_kde
import scipy.integrate as integrate
from operator import itemgetter
from datetime import datetime
from matplotlib.ticker import NullFormatter
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import sklearn.datasets
#plotly.offline.init_notebook_mode(connected=True)

comp=2
feat=2


def likelihood(X,mean,cov,mix):
    # gmmorig=mixture.GaussianMixture(n_components=comp)
    # gmmorig.weights_=mix
    # gmmorig.covariances_=cov
    # gmmorig.means_=mean
    # gmmorig.precisions_cholesky_=mixture.gaussian_mixture._compute_precision_cholesky(cov, 'full')
    # return gmmorig.lower_bound_
    like = 1


    like_new = np.array([0 for i in range(len(X))],dtype=float);
    for cmp in range(comp):
        like_new += stats.multivariate_normal.pdf(X[i],mean[cmp],cov[cmp])*mix[cmp];
    #print(like_new);
    like = np.exp(np.sum(np.log(like_new)));
    return like;



def gmm_kl(gmm_p, gmm_q, n_samples=10**4):
    X = gmm_p.sample(n_samples)
    try:    
        log_p_X = gmm_p.score_samples(X[0])
        log_q_X = gmm_q.score_samples(X[0])
    except Exception as e:
        print(e)
        print(X[0])
    return log_p_X.mean() - log_q_X.mean()

def compute_KL_divergence(mean,cov,mix,estimate):
    gmmorig=mixture.GaussianMixture(n_components=comp)
    gmmorig.weights_=mix
    gmmorig.covariances_=cov
    gmmorig.means_=mean
    gmmorig.precisions_cholesky_=mixture.gaussian_mixture._compute_precision_cholesky(cov, 'full')
    return gmm_kl(gmmorig,estimate)

        
def print_result(algo,lik,loc,scale,mix,t=-1):
    # if(t!=-1):
    #     print("Time: ",t);
    print(algo)
    
    print("Likelihood value:",lik);
    #print("Number of components:",comp)
    print("Means:",loc);
    print("Variance:",scale)
    print("Mixing parameters:",mix);
    

    
def k_means():   
    clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',max_iter=500,tol=10**-10).fit(np.array(num))
    return clf;


def greedy_method():
    clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',max_iter=1,tol=10**-10).fit(np.array(num))
    clfmaxmax=clf;
    clfmax=clf;
    mean=np.array(np.eye(comp,feat));
    precisions = [[[0 for a in range(feat)] for b in range(feat)] for c in range(comp)]
    maximum1=-1000000000000
    for i in range(10):
        maximum2=-1000000000000
        for j in range(10):
            labels=clfmaxmax.predict(num)
            post_prob = clfmaxmax.predict_proba(num)
            post_prob_t = post_prob.T
            for cmp in range(comp):
                nkk = np.sum(post_prob_t[cmp])
                # nk=np.sum(np.array(labels==cmp,dtype=int))
                # if(nk == 0):
                #     continue
                mean[cmp]= np.random.multivariate_normal(mean=clfmaxmax.means_[cmp],cov=clfmaxmax.covariances_[cmp]/nkk)
                matrix = np.array([[0 for a in range(feat)] for b in range(feat)], dtype=float)
                for d in range(len(num)):
                    arr = np.reshape(np.array(num[d]-mean[cmp]),(1,feat))
                    matrix += post_prob[d][cmp]*np.matmul(np.transpose(arr),arr)

                precisions[cmp] = np.linalg.inv(np.array(matrix/nkk))
            precisions = np.array(precisions)

            clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array(mean), precisions_init=np.array(precisions), weights_init=clfmaxmax.weights_, max_iter=10,tol=10**-10).fit(np.array(num))
            
            # clf=mixture.GaussianMixture(n_components=comp)
            # clf.weights_ = clfmaxmax.weights_
            # clf.covariances_ = np.array(precisions)
            # clf.means_ = np.array(mean)
            # clf.precisions_cholesky_ = mixture.gaussian_mixture._compute_precision_cholesky(precisions, 'full')
            # clf.fit(np.array(num))


            if(clf.lower_bound_>maximum2):
                maximum2 = clf.lower_bound_
                clfmax = clf

        if(clfmax.lower_bound_>maximum1):
            maximum1 = clfmax.lower_bound_
            clfmaxmax = clfmax
    #clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array(mean),max_iter=100,tol=10**-10).fit(np.array(num))
    clf = clfmaxmax
    return clf;



def rand_method():
    clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',max_iter=1,tol=10**-10).fit(np.array(num))
    clfmin=clf;
    mean=np.array(np.eye(comp,feat));
    precisions = [[[0 for a in range(feat)] for b in range(feat)] for c in range(comp)]
    maximum=-1000000000000
    for i in range(40):
        labels=clfmin.predict(num)
        post_prob = clfmin.predict_proba(num)
        post_prob_t = post_prob.T
        for cmp in range(comp):
            # nk=np.sum(np.array(labels==cmp,dtype=int))
            nkk = np.sum(post_prob_t[cmp])
            # if(nk == 0):
            #    continue
            mean[cmp] = np.random.multivariate_normal(mean=clfmin.means_[cmp],cov=clfmin.covariances_[cmp]/nkk)
            matrix = np.array([[0 for a in range(feat)] for b in range(feat)], dtype=float)
            for d in range(len(num)):
                arr = np.reshape(np.array(num[d]-mean[cmp]),(1,feat))
                matrix += post_prob[d][cmp]*np.matmul(np.transpose(arr),arr)

            precisions[cmp] = np.linalg.inv(np.array(matrix/nkk))
        precisions = np.array(precisions)

        clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array(mean), precisions_init=np.array(precisions), weights_init=clfmin.weights_, max_iter=10,tol=10**-10).fit(np.array(num))
        
        # clf=mixture.GaussianMixture(n_components=comp)
        # clf.weights_ = clfmin.weights_
        # clf.covariances_ = np.array(precisions)
        # clf.means_ = np.array(mean)
        # clf.precisions_cholesky_ = mixture.gaussian_mixture._compute_precision_cholesky(precisions, 'full')
        # clf.fit(np.array(num))


        if(clf.lower_bound_>maximum):
            maximum = clf.lower_bound_
            clfmin = clf
    clf = clfmin
    #clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array(mean),max_iter=100,tol=10**-10).fit(np.array(num))
    return clf;


def rand_sa():
    clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',max_iter=1,tol=10**-10).fit(np.array(num))
    #clfmin=clf;
    prev=clf;
    temp=1;
    minimum=1000000000000
    for i in range(6):
        mean = np.random.normal(loc=prev.means_,scale=prev.covariances_.reshape(prev.means_.shape)**0.5)
        #mean2 = np.random.normal(loc=clf.means_[1],scale=clf.covariances_[1][0]**0.5)
        #mean1 = np.random.uniform(xmin,xmax)
        #mean2 = np.random.uniform(xmin,xmax)
        newclf = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array(mean),max_iter=10,tol=10**-10).fit(np.array(num))
        if(newclf.lower_bound_<prev.lower_bound_):
            if(i==20):
                break;
                
            delta=prev.lower_bound_-newclf.lower_bound_;
            prob=np.exp(-delta/temp);
            check=np.random.uniform(0,1);
            if(check<prob):
                prev=newclf;
                #clf=newclf;
        else:
            prev=newclf;
            #clf=newclf;
        temp-=0.005*temp;        

    return prev;
    


def greedy_sa():
    clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',max_iter=1,tol=10**-10).fit(np.array(num))
    prev=clf;
    clfmax=clf;
    temp = 1
    maximum=-1000000000000
    for i in range(10):
        for j in range(5):
            mean = np.random.normal(loc=prev.means_,scale=prev.covariances_.reshape(prev.means_.shape)**0.5)
            #mean2 = np.random.normal(loc=clf.means_[1],scale=clf.covariances_[1][0]**0.5)
            #mean1 = np.random.uniform(xmin,xmax)
            #mean2 = np.random.uniform(xmin,xmax)
            clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array(mean),max_iter=10,tol=10**-10).fit(np.array(num))
            if(clfmax.lower_bound_<clf.lower_bound_):
                clfmax=clf;
        if(clfmax.lower_bound_<prev.lower_bound_):
            if(i==5):
                break;
            delta=prev.lower_bound_-clfmax.lower_bound_;
            prob=np.exp(-delta/temp);
            check=np.random.uniform(0,1);
            if(check<prob):
                prev=clfmax;
        else:
            prev = clfmax;
        temp -= 0.2*temp;   
    
    return prev;   
    
    


for tries in range(5):
    plt.figure() 
    num=[]
    mix=np.random.uniform(0,1,comp) # np.array([0.5,0.3,0.2])
    mix = np.array([0.6194, 0.3806])
    mix/=(sum(mix))
    loc,scale=np.random.uniform(-2,2,(feat,comp)),np.random.uniform(0,2,comp)#np.array([0,1,-1]),np.array([12,0.6,3])
    loc = np.array([[1.2873, 0.2989],[1.1834, 1.9912]])
    cov=[]
    for i in range(comp):
        cov.append(sklearn.datasets.make_spd_matrix(feat))
    cov=np.array(cov)
        
    cov = np.array([[[2.5592, -0.6467],[-0.6467, 1.0216]],[[1.7265, -1.0236],[-1.0236, 1.3155]]])

    for i in range(10000):
        j = np.random.choice(np.arange(0,comp),p=[x for x in mix])
        num.append(np.random.multivariate_normal(mean=loc[j],cov=cov[j]))


    
    num = np.array(num)
    plt.hist(num, bins=100,alpha=0.6, normed=True)
    xmin, xmax = plt.xlim()
    x_grid = np.linspace(xmin,xmax,150)
    x = x_grid.reshape(-1, 1)
    print_result("Actual values",likelihood(num,loc,cov,mix),loc,cov,mix,0);

   
    t1=time.time();
    clf1=k_means();
    t2=time.time();
    print_result("K-means",np.exp(clf1.lower_bound_),clf1.means_,clf1.covariances_,clf1.weights_,t2-t1)
    l = []   
    zzz=[]
    for x1 in x_grid:
        zz=[]
        for x2 in x_grid:
            l.append([x1,x2])
            temp=(np.exp(clf1.score_samples(np.array([[x1,x2]]))));
            zz.append(float(temp));
        zzz.append(zz);
    zzz=np.array(zzz);
    plt.figure()
    plt.contour(x_grid,x_grid,zzz,20);
    plt.savefig("K-MEANS Contour " + str(tries) + ".png")

    
    t1=time.time();
    clf2=rand_method()
    t2=time.time();
    print_result("Random",np.exp(clf2.lower_bound_),clf2.means_,clf2.covariances_,clf2.weights_,t2-t1)
    l = []   
    zzz=[]
    for x1 in x_grid:
        zz=[]
        for x2 in x_grid:
            l.append([x1,x2])
            temp=(np.exp(clf2.score_samples(np.array([[x1,x2]]))));
            zz.append(float(temp));
        zzz.append(zz);
    zzz=np.array(zzz);
    plt.figure()
    plt.contour(x_grid,x_grid,zzz,20);
    plt.savefig("RANDOM Contour " + str(tries) + ".png")
    
    
    t1=time.time();
    clf3=greedy_method();
    t2=time.time();
    print_result("Greedy",np.exp(clf3.lower_bound_),clf3.means_,clf3.covariances_,clf3.weights_,t2-t1)
    l = []   
    zzz=[]
    for x1 in x_grid:
        zz=[]
        for x2 in x_grid:
            l.append([x1,x2])
            temp=(np.exp(clf3.score_samples(np.array([[x1,x2]]))));
            zz.append(float(temp));
        zzz.append(zz);
    zzz=np.array(zzz);
    plt.figure()
    plt.contour(x_grid,x_grid,zzz,20)
    plt.savefig("GREEDY Contour " + str(tries) + ".png")


    print("---------K-L Divergence values-----------")
    print("1. K-MEANS ")
    print(compute_KL_divergence(loc,cov,mix,clf1))
    print("2. RANDOM ")
    print(compute_KL_divergence(loc,cov,mix,clf2))
    print("3. GREEDY ")
    print(compute_KL_divergence(loc,cov,mix,clf3))
    print("-------------------------------")


# plt.show()


    