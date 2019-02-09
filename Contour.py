import numpy as np
x_grid = np.linspace(-4.3,7.4,100)
import matplotlib.pyplot as plt
y_grid=x_grid
from scipy.stats  import multivariate_normal
mu1 = np.array([1.24135797, 0.26783094]);
mu2= np.array( [1.1969804 , 1.91507089])
sigma1 = np.array([[2.57804753, -0.65054746], [-0.65054746,  0.99427764]])
sigma2 = np.array( [ [1.80285616, -1.0791087],[-1.0791087 ,1.41316807]])
pi=np.array([0.58569045, 0.41430955])
zz=[]
for x in x_grid:
    z=[]
    for y in y_grid:
                
        z.append(pi[0]*multivariate_normal.pdf(np.array([[x,y]]),mu1,sigma1)+pi[1]*multivariate_normal.pdf(np.array([[x,y]]),mu2,sigma2));
    zz.append(z)
zz=np.array(zz)
plt.figure()
plt.contour(x_grid,y_grid,zz,20)
plt.savefig("ORIGINAL_Contour.png")