
# coding: utf-8

# In[101]:


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

comp=10
feat=20

#mle=-10000000;
#maxclf=mixture.GaussianMixture(n_components=comp,covariance_type='full',max_iter=1).fit(np.array(num));
#mle=maxclf.lower_bound_;
'''def rec(clf,depth):
    print("in loop");
    global mle
    global maxclf
    if(depth==5):
        return;
    mymeans=clf.means_.reshape(clf.weights_.shape);
    mycovars=clf.covariances_.reshape(clf.weights_.shape);
    randsamples=np.random.normal(loc=mymeans,scale=mycovars**0.5,size=(5,mymeans.shape[0]))
   
    #clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array(mean),max_iter=50,tol=10**-10).fit(np.array(num))
    #clf2=
    for x in randsamples:
        #clf.means_=x;
        clf2 = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array(x),max_iter=50,tol=10**-10).fit(np.array(num))# if(rec(clf,depth))
        rec(clf2,depth+1);
    if(clf.lower_bound_>mle):
        mle=clf.lower_bound_;/;
        maxclf=clf;
        
    return;  
'''    
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
    # print("Means:",loc);
    # print("Variance:",scale)
    # print("Mixing parameters:",mix);
    
    
def k_means():
    
    clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',max_iter=400,tol=10**-10).fit(np.array(num))
    return clf;


def greedy_method():
    clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',max_iter=1,tol=10**-10).fit(np.array(num))
    clfmaxmax=clf;
    clfmax=clf;
    mean=np.array(np.eye(comp,feat));
    maximum=-1000000000000
    for i in range(10):
        clfliks=[];
        clfprobs=[]
        for j in range(10):
            for cmp in range(comp):
                mean[cmp]= np.random.multivariate_normal(mean=clfmax.means_[cmp],cov=clfmax.covariances_[cmp])
            #mean2 = np.random.normal(loc=clf.means_[1],scale=clf.covariances_[1][0]**0.5)
            #mean1 = np.random.uniform(xmin,xmax)
            #mean2 = np.random.uniform(xmin,xmax)
            clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array(mean),max_iter=10,tol=10**-10).fit(np.array(num))
            clfliks.append(clf)
            clfprobs.append(np.exp(clf.lower_bound_))
        s=sum(clfprobs)
        for i in range(len(clfprobs)):
            clfprobs[i]/=s
        ind=np.random.choice(np.arange(0,len(clfliks)),p=clfprobs)
        clfmax=clfliks[ind]
        #Note , the following statement nullifies the above to revert to the old strategy
        clfmax=clfliks[clfprobs.index(max(clfprobs))]
        if(clfmax.lower_bound_>maximum):
            maximum = clf.lower_bound_#(np.array(num))
            clfmaxmax = clfmax
    #clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array(mean),max_iter=100,tol=10**-10).fit(np.array(num))
    clf = clfmaxmax
    return clf;


def rand_method():
    #print(num);
    clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',max_iter=1,tol=10**-10).fit(np.array(num))
    clfmin=clf;
    mean=np.array(np.eye(comp,feat));
   # print(clfmin.covariances_)
    minimum=1000000000000
    for i in range(40):
       # print(clfmin.means_)
        for cmp in range(comp):
            mean[cmp] = np.random.multivariate_normal(mean=clfmin.means_[cmp],cov=clfmin.covariances_[cmp])
        #mean2 = np.random.normal(loc=clf.means_[1],scale=clf.covariances_[1][0]**0.5)
        #mean1 = np.random.uniform(xmin,xmax)
        #mean2 = np.random.uniform(xmin,xmax)
        clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array(mean),max_iter=10,tol=10**-10).fit(np.array(num))
        if(clf.bic(np.array(num))<minimum):
            minimum = clf.bic(np.array(num))
            clfmin = clf
    clf = clfmin
    clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array(mean),max_iter=100,tol=10**-10).fit(np.array(num))
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
        #mean2 = np.random.normal(loc=clf.means_[1],scale=clf.covariances_[1][0]**0.5)
        #mean1 = np.random.uniform(xmin,xmax)
        #mean2 = np.random.uniform(xmin,xmax)
        clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array(mean),max_iter=10,tol=10**-10).fit(np.array(num))
        l.append(clf)
        if(clf.lower_bound_>minimum):
            minimum = clf.lower_bound_
            clfmin = clf
    clf = clfmin
    clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array(mean),max_iter=100,tol=10**-10).fit(np.array(num))
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
    
cntk=0;
cntrand=0;
cntg=0;
for tries in range(5):
    plt.figure()
    
    num=[]
    mix=np.random.uniform(0,1,comp) #np.array([0.5,0.3,0.2])
    mix/=(sum(mix))
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
        #print(loc[j])
        num.append(np.random.multivariate_normal(mean=loc[j],cov=cov[j]))
        #if(j == 0):
        #    num.append(np.random.normal(loc=loc1,scale=scale1))
       # else:
        #    num.append(np.random.normal(loc=loc2,scale=scale2))

    num = np.array(num)
    plt.hist(num, bins=100,alpha=0.6, normed=True)
    xmin, xmax = plt.xlim()
    x_grid = np.linspace(xmin,xmax,150)
    x = x_grid.reshape(-1, 1)
    print_result("Actual values",0,loc,scale*scale,mix,0);
   
    t1=time.time();
    clf1=k_means();
    t2=time.time();
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
    print("1. K-MEANS ")
    print(compute_KL_divergence(loc,cov,mix,clf1))
    print("2. RANDOM ")
    print(compute_KL_divergence(loc,cov,mix,clf2))
    print("3. GREEDY ")
    print(compute_KL_divergence(loc,cov,mix,clf3))
    print("-------------------------------")
    


    # plt.plot(x,x,np.exp(clf3.score_samples(x)))
    # plt.plot(x,np.exp(clf3.score_samples(x)),color='g')


    if(clf1.lower_bound_>clf2.lower_bound_ and clf1.lower_bound_>clf3.lower_bound_):
        cntk+=1;
    if(clf2.lower_bound_>clf1.lower_bound_ and clf2.lower_bound_>clf3.lower_bound_):
        cntrand+=1;
    if(clf3.lower_bound_>clf1.lower_bound_ and clf3.lower_bound_>clf2.lower_bound_):
        cntg+=1;
    
    
    #plt.show()

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
print("K-means count: ",cntk);
print("Rand count: ",cntrand);
print("Greedy count: ",cntg);
    
    


# In[2]:


