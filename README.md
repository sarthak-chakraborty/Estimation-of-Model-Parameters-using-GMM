> Work in progress  
> To be open sourced upon completion

# Estimation of Model Parameters using GMM

Expectation Maximization (EM) algorithm is one of the most popular iterative algorithms used in the parametric density estimation in statistics. The main aim in the parametric density estimation is to estimate the parameters of the specified density from the evidence by maximizing the likelihood function. Thus EM algorithm essentially seeks the maximum of the likelihood function. However, this algorithm leads to a local maximum of the likelihood function rather than the absolute maximum, depending on the initial guess for the parameters. This leads to the parameter estimates which do not represent the data accurately and still are used in subsequent analyses which may lead to undesirable decisions. In order to overcome this issue, we propose a variant of EM algorithm which outperforms the existing EM algorithm in case of parameter estimation of mixture Gaussian densities.

Hence we designed a randomized version of the EM algorithm where the steps of randomization are included in between the steps of successive EM algorithm to initialize it. This novel approach works better than the standard implementation and aims at reaching a better local maximum.


We implemented the code in python using the scikit-learn package. However the work is still in progress and hence unable to share the code at this point of time.


Here are some of the observations that we noticed.

![Contour Plot showing the superiority of our approach (RandEM & GreedEM) over standard approach (RegEM)](https://drive.google.com/open?id=1pLnIm_HBCuER5bbw1JB85xONm-_uOVDk)
