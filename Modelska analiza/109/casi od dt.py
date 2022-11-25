import random as rnd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.style.use('seaborn')
doLog = 0

beta = 0.5
beta_all = [0.1,0.25,0.5,1]
dt_all = np.linspace(0.1, 1, 10)
N = 250
dt = 0.1
iterations = 10000


for beta in beta_all:
    t_mean_all = []
    for dt in dt_all:
        t_mean = 0
        for i in range(iterations):
            N_now = N
            t_now = 0
            lambdaP_now = N_now*beta*dt
            while (N_now > 0):
                N_now = N_now - np.random.poisson(lambdaP_now)
                t_now = t_now + dt
                lambdaP_now = N_now*beta*dt

            t_mean += t_now/iterations
        t_mean_all.append(t_mean)
        
    plt.plot(dt_all, t_mean_all, label =  '$\u03B2 =$ ' + str(format(beta, '.2f')))
    
plt.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1)
plt.xlabel('$dt$')
plt.ylabel('$t_{death}$')
plt.title('$t_{death}(dt)$ - $N_0 = $ ' + str(N) + ', število ponovitev $= $' + str(iterations))

plt.show()
        
        




