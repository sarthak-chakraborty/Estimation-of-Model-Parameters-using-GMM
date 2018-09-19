# Estimation of Model Parameters using GMM

Expectation Maximization (EM) algorithm is one of the most popular iterative algorithms used in the parametric density estimation in statistics. The main aim in the parametric density estimation is to estimate the parameters of the specified density from the evidence by maximizing the likelihood function. Thus EM algorithm essentially seeks the maximum of the likelihood function. However, this algorithm leads to a local maximum of the likelihood function rather than the absolute maximum, depending on the initial guess for the parameters. This leads to the parameter estimates which do not represent the data accurately and still are used in subsequent analyses which may lead to undesirable decisions. In order to overcome this issue, we propose a variant of EM algorithm which outperforms the existing EM algorithm in case of parameter estimation of mixture Gaussian densities.

Hence our approach is to use a better algorithm for the initialization of the initial parameters of the EM algorithm so that the modified EM algorithm will converge to a better local maxima.


We implemented the code in python using the scikit-learn package. However the work is still in progress and hence unable to share the code at this point of time.


Here are some of the observations that we noticed.
