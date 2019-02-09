
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
import sklearn.datasets
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
#plotly.offline.init_notebook_mode(connected=True)

feat=28
comp=18


def checkno(s):
    try:
        float(s)
        return 1
    except:
        return 0


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
    return (gmm_kl(gmmorig,estimate)+gmm_kl(estimate,gmmorig))/2
        
def print_result(algo,lik,loc,scale,mix,t=-1):
    # if(t!=-1):
    #     print("Time: ",t);
    print(algo)
    
    print("Likelihood value:",lik);
    #print("Number of components:",comp)
    # print("Means:",loc);
    # print("Variance:",scale)
    # print("Mixing parameters:",mix);
    
    
def k_means():
    clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',max_iter=400,tol=10**-10).fit(np.array(num))
    return clf;


def rand2_method():
    clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',max_iter=1,tol=10**-10).fit(np.array(num))
    minimum=1000000000000
    k_mean=clf.means_
    #k_mean=clf.means_[1]
    var=clf.covariances_.reshape(clf.means_.shape)
    #var_2=clf.covariances_[1][0]
    var**=0.5;
    #var_2**=0.5
    for i in range(10):
        mean = np.random.normal(loc=k_mean,scale=var)
        #mean2 = np.random.normal(loc=k_mean2,scale=var_2)
        #mean1 = np.random.uniform(xmin,xmax)
        #mean2 = np.random.uniform(xmin,xmax)
        clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array(mean),max_iter=20,tol=10**-10).fit(np.array(num))
        if(clf.bic(np.array(num))<minimum):
            minimum = clf.bic(np.array(num))
            clfmin = clf
    clf = clfmin
    return clf;


def greedy_method():
    clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',max_iter=1,tol=10**-10).fit(np.array(num))
    clfmaxmax=clf;
    clfmax=clf;
    mean=np.array(np.eye(comp,feat));
    precisions = [[[0 for a in range(feat)] for b in range(feat)] for c in range(comp)]
    maximum1=-1000000000000
    for i in range(8):
        maximum2=-1000000000000
        for j in range(5):
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

def rand_rand_method():
    clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',max_iter=1,tol=10**-10).fit(np.array(num))
    clfmin=clf;
    mean=np.array(np.eye(comp,feat));
   # print(clfmin.covariances_)
    l=[];
    lprob=[];
    minimum=-1000000000000
    for i in range(40):
       # print(clfmin.means_)
        for cmp in range(comp):
            mean[cmp] = np.random.multivariate_normal(mean=clfmin.means_[cmp],cov=clfmin.covariances_[cmp])
        clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array(mean),max_iter=10,tol=10**-10).fit(np.array(num))
        l.append(clf)
        if(clf.lower_bound_>minimum):
            minimum = clf.lower_bound_
            clfmin = clf
    clf = clfmin
    clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array(mean),max_iter=100,tol=10**-10).fit(np.array(num))
    return clf;

      
    
cntk=0;
cntrand=0;
cntg=0;

for tries in range(1):
    plt.figure()
    
    num=[]

    mix=np.random.uniform(0,1,comp) #np.array([0.5,0.3,0.2])
    mix/=(sum(mix))
    print(mix)
    print(type(mix))
    loc,scale=np.random.uniform(-2,2,(comp,feat)),np.random.uniform(0,2,comp)#np.array([0,1,-1]),np.array([12,0.6,3])
    cov=[]
    for i in range(comp):
        cov.append(sklearn.datasets.make_spd_matrix(feat))
    cov=np.array(cov)
        
    #loc2,scale2=np.random.uniform(-1,1),np.random.uniform(0,5)
    #loc1,scale1=0,2
    #loc2,scale2=0.8,0.1
    #mix=0.5

    for i in range(10000):

        j = np.random.choice(np.arange(0,comp),p=[x for x in mix])
        num.append(np.random.multivariate_normal(mean=loc[j],cov=cov[j]))


    # df = pd.read_csv("/home/sarthak/Desktop/Shell Codes/segmentation.data",header=None)
    # df = df.iloc[:,17:20]
    # num = np.array(df)

    num = np.array(num)
    plt.hist(num, bins=100,alpha=0.6, normed=True)
    xmin, xmax = plt.xlim()
    x_grid = np.linspace(xmin,xmax,150)
    x = x_grid.reshape(-1, 1)
    # print_result("Actual values",0,loc,scale*scale,mix,0);
   
    t1=time.time();
    clf1=k_means();
    t2=time.time();
    cntk += np.exp(clf1.lower_bound_)
    print_result("K-means",np.exp(clf1.lower_bound_),clf1.means_,clf1.covariances_,clf1.weights_,t2-t1)
    '''
    print("time for K_means",t2-t1);
    print("k_means",np.exp(clf1.lower_bound_));
    print("k_means means- ",clf1.means_)
    print("k_means variances- ",clf1.covariances_)
    print("k_means mixing parameters",clf1.weights_)
    '''
  #  plt.plot(x,np.exp(clf1.score_samples(x)),color='r');
    
    t1=time.time();
    clf2=rand_method()
    t2=time.time();
    cntrand += np.exp(clf2.lower_bound_)
    '''
    print("time for random",t2-t1);
    print("random",np.exp(clf2.lower_bound_));
    print("random means ",clf2.means_)
    print("random covariances ",clf2.covariances_)
    print("random mixing parameters",clf2.weights_)
    '''
    print_result("Random",np.exp(clf2.lower_bound_),clf2.means_,clf2.covariances_,clf2.weights_,t2-t1)
  #  plt.plot(x,np.exp(clf2.score_samples(x)),color='k')
    
    
    t1=time.time();
    clf3=greedy_method();
    t2=time.time();
    cntg += np.exp(clf3.lower_bound_)
    '''
    print("time for greedy",t2-t1);
    print("greedy",np.exp(clf3.lower_bound_));
    print("greedy means ",clf3.means_
    print("greedy covariances ",clf3.covariances_)
    print("greedy mixing parameters",clf3.weights_)
    #print(np.exp(clf.lower_bound_))
    '''
    print_result("Greedy",np.exp(clf3.lower_bound_),clf3.means_,clf3.covariances_,clf3.weights_,t2-t1)
    
    print("---------K-L Divergence values-----------")
    print("1. K-MEANS \n")
    print(compute_KL_divergence(loc,cov,mix,clf1))
    print("2. RANDOM ")
    print(compute_KL_divergence(loc,cov,mix,clf2))
    print("3. GREEDY ")
    print(compute_KL_divergence(loc,cov,mix,clf3))
    print("-------------------------------")
    
    # plt.plot(x,x,np.exp(clf3.score_samples(x)))
    # plt.plot(x,np.exp(clf3.score_samples(x)),color='g')

    # if(clf1.lower_bound_>clf2.lower_bound_ and clf1.lower_bound_>clf3.lower_bound_):
    #     cntk+=1;
    # if(clf2.lower_bound_>clf1.lower_bound_ and clf2.lower_bound_>clf3.lower_bound_):
    #     cntrand+=1;
    # if(clf3.lower_bound_>clf1.lower_bound_ and clf3.lower_bound_>clf2.lower_bound_):
    #     cntg+=1;
    
    


    '''
    t1=time.time();
    clf4=rand_sa()
    t2=time.time();
    
    print("time for random_sa",t2-t1);
    print("random_sa",np.exp(clf4.lower_bound_));
    print("random_sa means ",clf4.means_)
    print("random_sa covariances ",clf4.covariances_)
    print("random_sa mixing parameters",clf4.weights_)
    
    print_result("Random SA",np.exp(clf4.lower_bound_),clf4.means_,clf4.covariances_,clf4.weights_,t2-t1)
    plt.plot(x,np.exp(clf4.score_samples(x)),color='k')
    
    
    t1=time.time();
    clf5=greedy_sa();
    t2=time.time();
    '''
    '''
    print("time for greedy_sa",t2-t1);
    print("greedy_sa",np.exp(clf5.lower_bound_));
    print("greedy_sa means ",clf5.means_)
    print("greedy_sa covariances ",clf5.covariances_)
    print("greedy_sa mixing parameters",clf5.weights_)
    
    print_result("Greedy SA",np.exp(clf5.lower_bound_),clf5.means_,clf5.covariances_,clf5.weights_,t2-t1)
    plt.plot(x,np.exp(clf5.score_samples(x)),color='g')
    '''
    '''    
    trace = go.Table(
        header=dict(values=['Algorithm', 'Observed Mean','Observed Variance', 'Observed Mixing Parameters','Likelihood Value']),
        cells=dict(values=[['K-Means', 'Random', 'Greedy', 'Random_sa', 'Greedy_sa'],
                           [clf1.means_, clf2.means_, clf3.means_, clf4.means_, clf5.means_],
                           [clf1.covariances_, clf2.covariances_, clf3.covariances_, clf4.covariances_, clf5.covariances_],
                           [clf1.weights_, clf2.weights_, clf3.weights_, clf4.weights_, clf5.weights_],
                           [np.exp(clf1.lower_bound_), np.exp(clf2.lower_bound_), np.exp(clf3.lower_bound_), np.exp(clf4.lower_bound_), np.exp(clf5.lower_bound_)]]))
    
    datatable = [trace]
    layout = dict(geo = {'scope':'usa'})

    choromap = go.Figure(data = [datatable],layout = layout)

    plotly.offline.plot(choromap)
    #py.iplot(datatable, filename = 'basic_table')

    # rec(maxclf,0)
    #plt.plot(x,np.exp(maxclf.score_samples(x)),color='g');
    #print("greedy",np.exp(clf.lower_bound_))
    
    a = -1
    num =[]
    num1 = []
    i=100000
    while(i):
        j = np.random.normal(loc=0.0,scale=1.0)
        num1.append(j)
        k = np.random.choice(np.arange(0,2),p=[0.5,0.5])
        if((j>a and k == 1) or (j<a and k==0)):
            num.append(j)

        i = i-1

    num2 = []
    num1.sort()
    for i in range(len(num1)):
        k = np.random.choice(np.arange(0,2),p=[0.5,0.5])
        if((num1[i]>a and k==1) or (num1[i]<a and k==0)):
            num2.append(num1[i])

    plt.hist(num,bins=100)
    plt.figure()
    plt.hist(num2,bins=100,normed=True)
    x1,x2=plt.xlim();
    plt.plot(np.linspace(x1,x2,100),norm.pdf(np.linspace(x1,x2,100),0,1));

    plt.show()
 
   '''

print("###########################")
print("Average")

print("K-means count: ",cntk/15);
print("Rand count: ",cntrand/15);
print("Greedy count: ",cntg/15);

#plt.show()   
