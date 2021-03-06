# coding: utf-8

# In[101]:



import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
from numpy import array
import numpy as np
import math
from scipy.stats import norm
import colorsys
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

comp=3
feat=2



import colorsys

from scipy.stats import multivariate_normal
import scipy.stats



def find_resp(data,obj):
    weights=obj.weights_
    cov=obj.covariances_
    mean=obj.means_
    resp=[]
    for x in data:
        var=[]
        for i in range(comp):
            var.append(weights[i]*multivariate_normal.pdf(x,mean[i],cov[i]))
        var/=sum(var)
        var=np.array(var)
#         var2 = multivariate_normal.pdf(x,mean[1],cov[1])
#         var3=var1*weights[0];
#         var3/=(var3+var2*weights[1])
        resp.append(var)
    resp=np.array(resp)
    # print(resp)
    return resp

def color(data,obj):
    N = comp
    HSV = [(float(x)/N, 0.75, 1) for x in range(1,N+1)]
    RGB = map(lambda x: colorsys.hsv_to_rgb(*x), HSV)
    # ans=find_resp(data,obj)
    predicted = obj.predict(data)
    # print(predicted)
    ans = [[0 for i in range(N)] for j in range(len(data))]
    for row in range(len(predicted)):
        ans[row][predicted[row]] += 1
    # print(ans)
    ct1=0
    ct2=0
    ct3=0
    for x in ans:
        if(x[0]>x[1] and x[0]>x[2]):
            ct1+=1;
        elif(x[1]>x[0] and x[1]>x[2]):
            ct2+=1;
        elif (x[2]>x[0] and x[2]>x[1]):
            ct3+=1;
    print(ct1,ct2,ct3)
    rgbmat=ans@np.array(list(RGB))
    data_col=data.T
    plt.scatter(data_col[0],data_col[1],c=rgbmat, s=20)

    
def data_gen():
    obj = mixture.GaussianMixture(n_components=2)
    obj.weights_=np.array([0.5,0.5])
    obj.means_ = np.array([[0.1,0.9],[2.3,1.8]])
    obj.covariances_=np.array([sklearn.datasets.make_spd_matrix(2), sklearn.datasets.make_spd_matrix(2)])
    obj.precisions_cholesky_=mixture.gaussian_mixture._compute_precision_cholesky(obj.covariances_,'full')
    
    return obj.sample(1000)[0],obj

# data,obj=data_gen()
# color(data,obj)

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
#     if(t!=-1):
#         print("Time: ",t);
    print(algo)
    
    print("Likelihood value:",lik);
    #print("Number of components:",comp)
#     print("Means:",loc);
#     print("Variance:",scale)
#     print("Mixing parameters:",mix);
    
    
def k_means():
    
    clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',max_iter=400).fit(np.array(num))
    return clf;

def rand2_method():
    clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',max_iter=1).fit(np.array(num))
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
        clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array(mean),max_iter=20).fit(np.array(num))
        if(clf.bic(np.array(num))<minimum):
            minimum = clf.bic(np.array(num))
            clfmin = clf
    clf = clfmin
    return clf;

def greedy_method():
    clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',max_iter=1).fit(np.array(num))
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
            clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array(mean),max_iter=10).fit(np.array(num))
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
    clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',max_iter=1).fit(np.array(num))
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
        clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array(mean),max_iter=10).fit(np.array(num))
        if(clf.bic(np.array(num))<minimum):
            minimum = clf.bic(np.array(num))
            clfmin = clf
    clf = clfmin
    clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array(mean),max_iter=100).fit(np.array(num))
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
iris = sklearn.datasets.load_iris()
X=iris.data
Y=X[:,:2]
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
        #print(loc[j])
        num.append(np.random.multivariate_normal(mean=loc[j],cov=cov[j]))
        #if(j == 0):
        #    num.append(np.random.normal(loc=loc1,scale=scale1))
       # else:
        #    num.append(np.random.normal(loc=loc2,scale=scale2))

    num = Y
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
    print("1. K-MEANS")
    print(compute_KL_divergence(loc,cov,mix,clf1))
    print("2. RANDOM ")
    print(compute_KL_divergence(loc,cov,mix,clf2))
    print("3. GREEDY ")
    print(compute_KL_divergence(loc,cov,mix,clf3))
    print("-------------------------------")
    
#     plt.plot(x,x,np.exp(clf3.score_samples(x)))
#     plt.plot(x,np.exp(clf3.score_samples(x)),color='g')
    if(clf1.lower_bound_>clf2.lower_bound_ and clf1.lower_bound_>clf3.lower_bound_):
        cntk+=1;
    if(clf2.lower_bound_>clf1.lower_bound_ and clf2.lower_bound_>clf3.lower_bound_):
        cntrand+=1;
    if(clf3.lower_bound_>clf1.lower_bound_ and clf3.lower_bound_>clf2.lower_bound_):
        cntg+=1;
        
    plt.figure()
    print(num)
    color(num,clf1)
    
    plt.figure()
    color(num,clf2)
    
    plt.figure()
    color(num,clf3)
    
    
    plt.show()



print("K-means count: ",cntk);
print("Rand count: ",cntrand);
print("Greedy count: ",cntg);
    
    


# In[2]: