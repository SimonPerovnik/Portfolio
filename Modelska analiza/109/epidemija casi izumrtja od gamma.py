import random as rnd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from scipy.sparse import diags

matplotlib.style.use('seaborn')
PlotT = 1

N = 500
N_all = [25,100,500,2000]
O0 = int(N*0.15)
I0 = int(N*0.01)
D0 = N - O0 - I0

alpha = 0.005
beta = 0.5
gamma = 0.003
gamma_all = [0.01,0.001,0.0001,0.000001]
dt = 0.5
iterations = 10000




numBin = 500
tBin_max = 35

for gamma in gamma_all:
    t_hist = []
    t_mean = 0
    for i in range(iterations):
        O0 = int(N*0.15)
        I0 = int(N*0.01)
        D0 = N - O0 - I0
        O = O0
        I = I0
        D = D0
        t_now = 0
        while( (O > 0) ):
            t_now = t_now + dt
            if t_now > tBin_max:
                #print('o ou')
                break
            D_new = (D  - np.random.poisson(alpha*D*O*dt) 
                        + np.random.poisson(gamma*I*dt)) 
            O_new = (O  + np.random.poisson(alpha*D*O*dt) 
                        - np.random.poisson(beta*O*dt))
            I_new = (I  + np.random.poisson(beta*O*dt) 
                        - np.random.poisson(gamma*I*dt))
            D = D_new
            O = O_new
            I = I_new
            # Make sure that number of not yet sick and immune people doesn't go below zero
            if (I<0): I = 0.1
            if (D<0): D = 0.1

        t_death = t_now
        t_mean +=t_death/iterations
        t_hist.append(t_now)
    hist, bins = np.histogram(t_hist, bins = numBin, range = (0,tBin_max))
    plt.bar(bins[1:],hist,
              label =  '$\u03B3 = $' + str(gamma) + ', $t_{mode} = $' + str(round(bins[np.argmax(hist)],1)) + ', $t_{mean} = $' + str(round(t_mean,1)),
              alpha = 0.6)

plt.title('Epidemija, $\u03B1 = $' + str(alpha) + ', $\u03B2 = $' + str(beta)+ '$N= $' + str(N) + ', $D_0 = $' + str(D0/N*100) 
          + '%, $O_0 = $' + str(O0/N*100) + '%, $I_0 = $' + str(I0/N*100) +  '%, \n$dt = $' + str(dt) + ', število iteracij $=$ ' + str(iterations))
plt.xlabel('$t_{end}$')
plt.ylabel('Število dogodkov')
plt.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1)
plt.show()