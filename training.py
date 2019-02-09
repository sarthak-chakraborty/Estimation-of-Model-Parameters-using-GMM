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

components = 2
data = []
timestamp = []
name = []
colors = ['b','r','g','y','c','m','k']
startingCol = 1
no_of_col = 1
alpha = 90


def checkno(s):
	try:
		float(s)
		return 1
	except:
		return 0


def kmeans(k):
	clf = mixture.GaussianMixture(n_components=components,covariance_type='full',max_iter=50).fit(k.reshape(-1,1))
	for i in range(9):
		#print(np.exp(clf.lower_bound_))
		clf = mixture.GaussianMixture(n_components=components,means_init=clf.means_,precisions_init=clf.precisions_,weights_init=clf.weights_,covariance_type='full',max_iter=50).fit(k.reshape(-1,1))

	return clf


def randomMethod(k):
	clf = mixture.GaussianMixture(n_components=components,covariance_type='full',max_iter=1).fit(k.reshape(-1,1))
	obj = clf
	maximum = -10000000000000

	for i in range(25):
		#print(np.exp(obj.lower_bound_))
		mean = np.random.normal(loc=obj.means_,scale=obj.covariances_.reshape(obj.means_.shape)**0.5)
		clf = mixture.GaussianMixture(n_components=components,covariance_type='full',means_init=np.array(mean).reshape(-1,1),max_iter=20).fit(k.reshape(-1,1))

		if(clf.lower_bound_> maximum):
			maximum = clf.lower_bound_
			obj = clf
	return obj


def greedyMethod(k):
	clf = mixture.GaussianMixture(n_components=components,covariance_type='full',max_iter=1).fit(k.reshape(-1,1))
	obj = clf
	maximum1 = -10000000000000
	for i in range(10):
		#print(np.exp(obj.lower_bound_))
		maximum2 = -10000000000000000
		for j in range(5):
			mean = np.random.normal(loc = obj.means_,scale=obj.covariances_.reshape(obj.means_.shape)**0.5)
			clf = mixture.GaussianMixture(n_components=components,covariance_type='full',means_init=np.array(mean).reshape(-1,1),max_iter=10).fit(k.reshape(-1,1))

			if(clf.lower_bound_ > maximum2):
				maximum2 = clf.lower_bound_
				obj1 = clf
		clf = obj1
		if(clf.lower_bound_ > maximum1):
			maximum1 = clf.lower_bound_
			obj = clf
	return obj


def getdata():
	counter = 3 + 2*(startingCol-1)
	for index in range(startingCol, startingCol+no_of_col):
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
				if(check<0.8):
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


def getdata():
	df = pd.read_csv("/home/sarthak/Desktop/Shell Codes/segmentation.data",header=None)
	global data
	data = df.iloc[:,1:19].as_matrix()


getdata()
data2=[]
data2.append((data,data));
data=np.array(data2)

getdata()
counter = 0
likelihood = []
likelihoodcount = {"kmeans":0 , "random":0 , "greedy":0}
for (k,testdata) in data:

	#HISTOGRAM PLOT
    plt.figure()
    plt.hist(k.astype('float'),bins=100,alpha=0.6,normed=True,color=colors[counter%7])
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.xlabel('Value Observed')
    plt.ylabel('Frequency')
    x_grid = np.linspace(xmin,xmax,100)
    x = x_grid.reshape(-1, 1)
    counter=counter+1


    #MIXTURE GAUSSIAN PLOT
    print(str(counter) + ": KMEANS")
    clf = kmeans(k)
    kmeanslikelihood = np.exp(clf.lower_bound_)
    print(np.exp(clf.lower_bound_))
    gmm_pdf = np.exp(clf.score_samples(x))
    #plt.plot(x_grid, gmm_pdf, 'k', linewidth=1)

    mix = clf.weights_
    means = clf.means_.reshape(mix.shape)
    covars = clf.covariances_.reshape(mix.shape)

    indexmin = min(enumerate(means), key=itemgetter(1))[0]
    indexmax = max(enumerate(means), key=itemgetter(1))[0]
    intervals=[]
    for i in range(len(means)):
        xx=means[i]
        yy=covars[i]
        yy=yy**0.5
        intervals.append((xx-2*yy,xx+2*yy)) 

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

    #plt.savefig("TimeSeriesKMeans" + str(counter) + ".png")

    print("RANDOM")
    clf = randomMethod(k)
    randomlikelihood = np.exp(clf.lower_bound_)
    print(np.exp(clf.lower_bound_))
    gmm_pdf = np.exp(clf.score_samples(x))
    #plt.plot(x_grid, gmm_pdf, 'r', linewidth=1)

    mix = clf.weights_
    means = clf.means_.reshape(mix.shape)
    covars = clf.covariances_.reshape(mix.shape)

    indexmin = min(enumerate(means), key=itemgetter(1))[0]
    indexmax = max(enumerate(means), key=itemgetter(1))[0]
    intervals=[]
    for i in range(len(means)):
        xx=means[i]
        yy=covars[i]
        yy=yy**0.5
        intervals.append((xx-2*yy,xx+2*yy)) 

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

    plt.rcParams.update({'font.size':12})


    #plt.savefig("TimeSeriesRandom" + str(counter) + ".png")

    print("GREEDY")
    clf = greedyMethod(k)
    greedylikelihood = np.exp(clf.lower_bound_)
    print(np.exp(clf.lower_bound_))
    gmm_pdf = np.exp(clf.score_samples(x))
    #plt.plot(x_grid, gmm_pdf, 'g', linewidth=1)

    # if(kmeanslikelihood > randomlikelihood):
    # 	if(kmeanslikelihood > greedylikelihood):
    # 		likelihood.append("kmeans")
    # 		likelihoodcount["kmeans"] += 1
    # 	else:
    # 		likelihood.append("greedy")
    # 		likelihoodcount["greedy"] += 1
    # else:
    # 	if(randomlikelihood > greedylikelihood):
    # 		likelihood.append("random")
    # 		likelihoodcount["random"] += 1
    # 	else:
    # 		likelihood.append("greedy")
    # 		likelihoodcount["greedy"] += 1


    mix = clf.weights_
    means = clf.means_.reshape(mix.shape)
    covars = clf.covariances_.reshape(mix.shape)

    indexmin = min(enumerate(means), key=itemgetter(1))[0]
    indexmax = max(enumerate(means), key=itemgetter(1))[0]
    intervals=[]
    for i in range(len(means)):
    	xx=means[i]
    	yy=covars[i]
    	yy=yy**0.5
    	intervals.append((xx-2*yy,xx+2*yy)) 
    '''
    warning=0
    warningdata = []
    for mydata in testdata:
    	l=[norm.pdf(mydata,means[i],covars[i]**0.5) for i in range(len(means))]
    	myind=max(enumerate(l), key=itemgetter(1))[0]
    	if(mydata<intervals[myind][0] or mydata>intervals[myind][1]):
    		warning+=1
    		warningdata.append(mydata)
    print("number of warnings out of 20,000 test data values %d (%f)" %(warning,float(warning)/(20000)*100))
    plt.figure()
    plt.hist(array(warningdata).astype('float'),bins=80)

    intervals = sorted(intervals)
    left = intervals[0][0]
    right = intervals[len(intervals)-1][1]
    plt.axvline(x=left, color='r', linestyle='--')
    plt.axvline(x=right, color='g', linestyle='--')
    print((left,right))
	
    plt.title(name[counter-1] + ", (L,R) = (" + str(left)[0:6] + ", " + str(right)[0:6] + ")")
    plt.xticks([left, right])
    '''

    #plt.savefig("Figure" + str(counter) + ".png")

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

    #plt.savefig("TimeSeriesGreedy" + str(counter) + ".png")

# print(likelihood)
# print(likelihoodcount)
plt.show()