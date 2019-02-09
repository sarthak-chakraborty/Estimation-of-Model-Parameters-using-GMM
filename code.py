
import csv
import pandas as pd
import matplotlib.pyplot as plt
from numpy import array
import numpy as np
import math
from scipy.stats import norm
import scipy.stats as stats
from sklearn import mixture
from sklearn.neighbors.kde import KernelDensity
from scipy.stats import gaussian_kde
import scipy.integrate as integrate
from operator import itemgetter
from datetime import datetime
from matplotlib.ticker import NullFormatter

nullfmt = NullFormatter()

def checkno(s):
    try:
        float(s)
        return 1;
    except:
        return 0;

#FIND AVERAGE OF TWO POINTS
def find_mid(low, high):
    mid = low + (high-low)/2
    return mid


#FIND THE PERCENT POINT BY INTEGRATING THE PDF
def cdf(p, xmin, xmax, x, alpha):
    pp = 1 - (1 - float(alpha)/100.0)/2
    i = (0.0 , 0.0)
    low = xmin
    high = xmax
    while(abs(i[0]-pp) > 0.000001):
        limu = find_mid(low, high)
        i = integrate.quad(lambda x: np.exp(p.score_samples(x)), xmin, limu)
        if(i[0] < pp):
            low = limu
        else:
            high = limu

    i = (0.0 , 0.0);
    low = xmin
    high = xmax
    while(abs(i[0]-pp) > 0.000001):
        liml = find_mid(low, high)
        i = integrate.quad(lambda x: np.exp(p.score_samples(x)), liml, xmax)
        if(i[0] < pp):
            high = liml
        else:
            low = liml

    return (liml, limu)

comp = 3

def kmeans(k):
    for i in range(10):
        clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',max_iter=20).fit(k.reshape(-1,1))
        #print(clf.lower_bound_)
    return clf



def greedyMethod(k):
    clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',max_iter=1).fit(k.reshape(-1,1))
    minimum = 100000000000000000
    number = 0

    for num in range(50):

        #print(num)
        #meansnew = clf.means_+(-1)**(np.random.choice(np.arange(0,2),p=[1,0]))*np.sqrt((clf.covariances_).reshape(clf.means_.shape))
        #meansnew = meansnew + np.random.uniform(0, 0.01)*meansnew
        maximum = -10000000000000
        for num1 in range(10):
            #print(num1)
            mean1=np.random.normal(loc=clf.means_[0],scale=clf.covariances_[0][0]**0.5)
            mean2=np.random.normal(loc=clf.means_[1],scale=clf.covariances_[1][0]**0.5)
            mean3=np.random.normal(loc=clf.means_[2],scale=clf.covariances_[2][0]**0.5)
            mean4=np.random.normal(loc=clf.means_[3],scale=clf.covariances_[3][0]**0.5)
            clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',weights_init=clf.weights_,precisions_init=clf.precisions_,means_init=np.array([mean1,mean2,mean3,mean4]).reshape(-1,1),max_iter=5).fit(k.reshape(-1,1))
            #clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array([mean1,mean2,mean3]).reshape(-1,1),max_iter=5).fit(k.reshape(-1,1))
            #gmm_pdf = np.exp(clf.score_samples(x))
            #plt.figure()
            #plot.hist(k.astype('float'),bins=100,alpha=0.6,normed=True,color=colors[counter%7])
            #plt.plot(x_grid, gmm_pdf, 'k', linewidth=1)
            #print(clf.bic(k.reshape(-1,1)))
        
            if(clf.score(k.reshape(-1,1)) > maximum):
                maximum=clf.score(k.reshape(-1,1))
                obj = clf
        clf = obj
        if(clf.bic(k.reshape(-1,1)) < minimum):
            minimum=clf.bic(k.reshape(-1,1))
            obj1 = clf

    return obj1



def randomMethod(k):
    clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',max_iter=1).fit(k.reshape(-1,1))
    minimum = 100000000000000000
    number = 0
    obj1 = clf
    for num in range(10):
         mean=np.random.normal(loc=obj1.means_,scale=obj1.covariances_.reshape(obj1.means_.shape)**0.5)
         #mean2=np.random.normal(loc=clf.means_[1],scale=clf.covariances_[1][0]**0.5)
         #mean3=np.random.normal(loc=clf.means_[2],scale=clf.covariances_[2][0]**0.5)
         #mean4=np.random.normal(loc=clf.means_[3],scale=clf.covariances_[3][0]**0.5)
         #clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',weights_init=clf.weights_,precisions_init=clf.precisions_,means_init=np.array([mean1,mean2,mean3]).reshape(-1,1),max_iter=10).fit(k.reshape(-1,1))
         clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array(mean).reshape(-1,1),max_iter=20).fit(k.reshape(-1,1))
        #gmm_pdf = np.exp(clf.score_samples(x))
        #plt.figure()
        #plot.hist(k.astype('float'),bins=100,alpha=0.6,normed=True,color=colors[counter%7])
        #plt.plot(x_grid, gmm_pdf, 'k', linewidth=1)
        #print(clf.bic(k.reshape(-1,1)))    
        
         if(clf.bic(k.reshape(-1,1)) < minimum):
            minimum=clf.bic(k.reshape(-1,1))
            obj1 = clf

    return obj1


def randomMethod2(k):
    clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',max_iter=1).fit(k.reshape(-1,1))
    minimum = 100000000000000000
    number = 0
    kmean = clf.means_
    kvar = clf.covariances_.reshape(clf.means_.shape)**0.5
    #kmean2 = clf.means_[1]
    #kvar2 = clf.covariances_[1][0]**0.5
    #kmean3 = clf.means_[2]
    #kvar3 = clf.covariances_[2][0]**0.5
    #kmean4 = clf.means_[3]
    #kvar4 = clf.covariances_[3][0]**0.5

    for num in range(10):
        #print(num)
        #meansnew = clf.means_+(-1)**(np.random.choice(np.arange(0,2),p=[1,0]))*np.sqrt((clf.covariances_).reshape(clf.means_.shape))
        #meansnew = meansnew + np.random.uniform(0, 0.01)*meansnew
        #print(num1)
         
         mean=np.random.normal(loc=kmean,scale=kvar)
         #mean2=np.random.normal(loc=kmean2,scale=kvar2)
         #mean3=np.random.normal(loc=kmean3,scale=kvar3)
         #mean4=np.random.normal(loc=kmean4,scale=kvar4)     
         #clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',weights_init=clf.weights_,precisions_init=clf.precisions_,means_init=np.array([mean1,mean2,mean3]).reshape(-1,1),max_iter=10).fit(k.reshape(-1,1))
         clf = mixture.GaussianMixture(n_components=comp,covariance_type='full',means_init=np.array(mean).reshape(-1,1),max_iter=20).fit(k.reshape(-1,1))
        #gmm_pdf = np.exp(clf.score_samples(x))
        #plt.figure()
        #plot.hist(k.astype('float'),bins=100,alpha=0.6,normed=True,color=colors[counter%7])
        #plt.plot(x_grid, gmm_pdf, 'k', linewidth=1)
        #print(clf.bic(k.reshape(-1,1)))
        
        
         if(clf.bic(k.reshape(-1,1)) < minimum):
            minimum=clf.bic(k.reshape(-1,1))
            obj1 = clf

    return obj1


colors = ['b','r','g','y','c','m','k']
data=[]    #ARRAY TO STORE THE DATA
timestamp=[]

#FOLLOWING BLOCK OF CODE EXTRACTS DATA FROM CSV FILE AND STORE IN data[] ARRAY
startingCol = 8
counter = 3 + 2*(startingCol-1)
name = []
for index in range(startingCol, startingCol+1):
    i=0
    x1 = []
    x2 = []
    y = []
    f = open("/home/sarthak/Desktop/G9801_GT/G-9801 GT.csv","r")
    reader = csv.reader(f)
    for row in reader:
        if(i == 0):
            name.append(row[counter])
        if(i>0):
            if(i == 100000):
                break
            check = np.random.uniform(0,1)
            if(check<1):
                if(checkno(row[counter])):
                    x1.append(float(row[counter]))
            else:
                if(checkno(row[counter])):
                    x2.append(float(row[counter]))
            y.append(row[counter-1])
        i = i+1

    f.close() 
    counter = counter + 2
    timestamp.append(y)
    data.append((array(x1),array(x2)))



'''
The following data prints the data in a histogram.
The X-Axis of the histogram is the value of the data points.
The Y-Axis of the histogram refers to the frequency of each data (more correctly, number of data points within the bin.
NORMAL, MIXTURE GAUSSIAN and KERNEL DENSITY can be fit to the obtained data plot.
'''
alpha = 90
counter=0
for (k,testdata) in data:
    
    #HISTOGRAM PLOT
    plt.figure()
    plt.hist(k.astype('float'),bins=200,alpha=0.6,normed=True,color=colors[counter%7])
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.xlabel('Value Observed')
    plt.ylabel('Frequency')
    x_grid = np.linspace(xmin,xmax,100)
    x = x_grid.reshape(-1, 1)
    counter=counter+1
    
    '''
    #NORMAL DISTRIBUTION PLOT
    mu,std = norm.fit(k)
    p = norm.pdf(x_grid, mu, std)
    plt.plot(x_grid, p, 'k', linewidth=1)
    title = "%d) Fit results: mu = %.2f,  std = %.2f" % (i, mu, std)
    plt.title(title) 
    '''

    
    #MIXTURE GAUSSIAN PLOT
    clf = kmeans(k)
    #clf = randomMethod(k)
    #clf = greedyMethod(k)
    
    print(clf.bic(k.reshape(-1,1)))

    mix=clf.weights_
    means = clf.means_
    covar = clf.covariances_
    covar_new = []
    means_new = []
    for j in means:
        means_new.append(float(j[0]))

    for j in covar:
        covar_new.append(float(j[0][0]))


    indexmin = min(enumerate(means_new), key=itemgetter(1))[0]
    indexmax = max(enumerate(means_new), key=itemgetter(1))[0]

    intervals=[]
    for i in range(len(means_new)):
        xx=means_new[i]
        yy=covar_new[i]
        yy=yy**0.5
        intervals.append((xx-2*yy,xx+2*yy))
    
    intervals = sorted(intervals)
    newalpha=100-(100-float(alpha))/float(mix[indexmin]+mix[indexmax]);
    
    gmm_pdf = np.exp(clf.score_samples(x))
    plt.plot(x_grid, gmm_pdf, 'k', linewidth=1)
    clf=randomMethod(k)
    print(clf.bic(k.reshape(-1,1)))
    gmm_pdf = np.exp(clf.score_samples(x))
    plt.plot(x_grid, gmm_pdf, 'r', linewidth=1)
    clf=randomMethod2(k)
    print(clf.bic(k.reshape(-1,1)))
    gmm_pdf = np.exp(clf.score_samples(x))
    plt.plot(x_grid, gmm_pdf, 'g', linewidth=1)
    liml1, limu1 = cdf(clf, xmin, xmax, x, 100-100*(1-float(newalpha)/100)*float(mix[indexmin]));
    liml2, limu2 = cdf(clf, xmin, xmax, x, 100-100*(1-float(newalpha)/100)*float(mix[indexmax]));
    #plt.axvline(x=liml1, color='k', linestyle='--')
    #plt.axvline(x=limu2, color='k', linestyle='--')
    
    #plt.axvline(x=intervals[0][0], color='r', linestyle='--')

    for i in range(0, len(intervals)-1):
        if(intervals[i][1] < intervals[i+1][0]):
            plt.axvline(x=intervals[i][1], color='r', linestyle='--');
            plt.axvline(x=intervals[i+1][0], color='g', linestyle='--')
    '''
    #plt.axvline(x=intervals[len(intervals)-1][1], color='g', linestyle='--')
    
    '''
    for i in range(0, len(intervals)):
        plt.axvline(x=intervals[i][0],color='r',linestyle='--')
        plt.text(intervals[i][0], ymax, "L"+str(i+1))
        plt.axvline(x=intervals[i][1],color='g',linestyle='--')
        plt.text(intervals[i][1], ymax, "R"+str(i+1))
    '''
    #print(cdf(clf, xmin, xmax, x, 100-100*(1-float(newalpha)/100)*float(mix[indexmin])));
    #print(cdf(clf, xmin, xmax, x, 100-100*(1-float(newalpha)/100)*float(mix[indexmax])))
    '''
    warning=0
    warningdata = []
    for mydata in testdata:
        l=[norm.pdf(mydata,means_new[i],covar_new[i]**0.5) for i in range(len(means_new))]
        myind=max(enumerate(l), key=itemgetter(1))[0]
        if(mydata<intervals[myind][0] or mydata>intervals[myind][1]):
            warning+=1
            warningdata.append(mydata)
    print("number of warnings out of 20,000 test data values %d (%f)" %(warning,float(warning)/(20000)*100))

    plt.title(name[counter-1] + " (L,R) = " + str(intervals[0][0]) + "," + str(intervals[len(intervals)-1][1]))
    plt.xticks([liml1, limu2])
    #plt.figure()
    #plt.hist(array(warningdata).astype('float'),bins=80)
    '''
    


    
    #TIME SERIES PLOT
    ts=np.array([datetime.strptime(timestamp[counter-1][val],"%d-%m-%Y %H:%M") for val in range(len(k))])
    plt.figure(figsize=(10,10))
    axmain = plt.axes((0.05,0.2,0.7,0.6))
    axhisty = plt.axes((0.8,0.2,0.15,0.6))
    axmain.plot(ts, k)
    #axmain.axhline(y=liml1, color='k', linestyle='--')
    #axmain.axhline(y=limu2, color='k', linestyle='--')
    
    axmain.axhline(y=intervals[0][0], color='r', linestyle='--')
    for i in range(0, len(intervals)):
        axmain.axhline(y=intervals[i][0], color='r', linestyle='--');
        axmain.axhline(y=intervals[i][1], color='g', linestyle='--')
    axmain.axhline(y=intervals[len(intervals)-1][1], color='g', linestyle='--')
    
    axhisty.hist(k.astype('float'),bins=100,alpha=0.6,normed=True,color=colors[(counter-1)%7],orientation='horizontal')
    #axhisty.axhline(y=liml1, color='k', linestyle='--')
    #axhisty.axhline(y=limu2, color='k', linestyle='--')
    
    axhisty.axhline(y=intervals[0][0], color='r', linestyle='--')
    for i in range(0, len(intervals)-1):
        if(intervals[i][1] < intervals[i+1][0]):
            axhisty.axhline(y=intervals[i][1], color='g', linestyle='--');
            axhisty.axhline(y=intervals[i+1][0], color='r', linestyle='--')
    axhisty.axhline(y=intervals[len(intervals)-1][1], color='g', linestyle='--')
    '''
    #plt.savefig("Histogram" + str(counter) + ".png")
    

    '''
    #KERNEL DENSITY PLOT (using scikit-learn)
    kde = KernelDensity(kernel='gaussian',bandwidth=0.01).fit(k.reshape(-1,1))
    kde_pdf = np.exp(kde.score_samples(x))
    plt.figure()
    plt.plot(x_grid, kde_pdf,'k',linewidth=1)
    liml, limu = cdf(kde, xmin, xmax, x, alpha)
    plt.axvline(x=liml, color='k', linestyle='--')
    plt.axvline(x=limu, color='k', linestyle='--')
    print(cdf(kde, xmin, xmax, x, alpha))
    '''


    '''
    #KERNEL DENSITY PLOT (using scipy)
    density = gaussian_kde(k)
    plt.plot(x_grid, density(x_grid), 'm', linewidth=1.2)
    '''

plt.show()