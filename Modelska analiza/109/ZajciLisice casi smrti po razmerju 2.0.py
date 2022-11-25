import random as rnd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from scipy.sparse import diags

matplotlib.style.use('seaborn')


Z0 = 200
L0 = 50
ratio = 1
ratio_all = [0.1,1.0,10]
alpha = 1
beta = alpha/ratio
dt = 0.01
iterations = 10000

numBin = 3000
tBin_max = 25
bins = np.linspace(0, tBin_max, numBin)

def getBin(value):
    for i in range(numBin):
        if value < bins[i]:
            return i-1
    return 0


for ratio in ratio_all:
    alpha = 1
    beta = alpha/ratio
    t_death_all = np.zeros(numBin)
    t_hist = []
    t_mean = 0
    print(ratio)
    for i in range(iterations):
        L = L0
        Z = Z0
        t_now = 0
        while( (L > 0) & (Z > 0)):
            t_now = t_now + dt
            if t_now > 50:
                break
            L_new = (L  + np.random.poisson(4*beta*L*dt) 
                    - np.random.poisson(5*beta*L*dt) 
                    + np.random.poisson(beta/Z0*Z*L*dt))
            Z_new = (Z  + np.random.poisson(5*alpha*Z*dt) 
                    - np.random.poisson(4*alpha*Z*dt) 
                    - np.random.poisson(alpha/L0*Z*L*dt))
            L = L_new
            Z = Z_new
        t_death = t_now

        t_mean +=t_death/iterations
        #t_death_all[getBin(t_death)] += 1
        t_hist.append(t_now)

    hist, bins = np.histogram(t_hist, bins = numBin, range = (0,tBin_max))
    t_death_all[0] = 0 
    plt.plot( bins[1:],hist,
              label =  '$\u03B1/\u03B2 = $' + str(ratio) + ', $t_{mode} = $' + str(round(bins[np.argmax(hist)],1)) + ', $t_{mean} = $' + str(round(t_mean,1)),
              alpha = 0.6)
plt.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1)
plt.xlabel('$t_{death}$')
plt.ylabel('$N$')
plt.title('Časi izumrtja - $Z_0 = $' + str(Z0) + ', $L_0 = $' + str(L0)  + ', $dt =$ ' + str(dt) + ', število ponovitev $= $' + str(iterations))

plt.show()