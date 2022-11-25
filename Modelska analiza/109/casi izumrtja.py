import random as rnd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.style.use('seaborn')
doLog = 0

beta = 0.5
beta_all = [0.1,0.2,0.3,0.4,0.5,0.75,1]
N = 250
dt = 0.1
iterations = 100000
i_max = 0

numBin = 3000
tBin_max = 90
bins = np.linspace(0, tBin_max, numBin)

def getBin(value):
    for i in range(numBin):
        if value < bins[i]:
            return i-1
    return 0


for beta in beta_all:
    t_death = np.zeros(numBin)
    t_mean = 0
    for i in range(iterations):
        N_t, t = [], []

        lambdaP_now = N*beta*dt
        N_now = N
        t_now = 0
        
        t.append(t_now)
        N_t.append(N_now)
        i = 1
        while (N_now > 0):
            N_now = N_now - np.random.poisson(lambdaP_now)
            t_now = t_now + dt
            lambdaP_now = N_now*beta*dt
            t.append(t_now)
            N_t.append(N_now)
            i = i+1
            
            if i > i_max:
                i_max = i
        t_mean += t_now/iterations
        t_death[getBin(t_now)] += 1
        
    t_death[0] = 0 
    plt.plot(bins, t_death,
              label =  '$\u03B2 =$ ' + str(format(beta, '.2f')) + ', $t_{mode} = $' + str(round(bins[np.argmax(t_death)],1)) + ', $t_{mean} = $' + str(round(t_mean,1)),
              alpha = 0.6)
plt.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1)
plt.xlabel('$t_{death}$')
plt.ylabel('$N$')
plt.title('Časi izumrtja, $N_0 = $ ' + str(N) + ', $dt =$ ' + str(dt) + ', število ponovitev $= $' + str(iterations))

plt.show()
        
        




