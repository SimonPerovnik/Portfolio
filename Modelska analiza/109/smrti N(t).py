import random as rnd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.style.use('seaborn')
doLog = 0

beta = 0.5
N = 25
dt = 0.1
iterations = 100
max = 400
N_mean = np.zeros(max)
t_mean = [dt *i for i in range(max)]
i_max = 0


for i in range(iterations):
    N_t, t = [], []

    lambdaP_now = N*beta*dt
    N_now = N
    t_now = 0

    t.append(t_now)
    N_t.append(N_now)
    N_mean[0] = N_now
    i = 1
    while (N_now > 0):
        N_now = N_now - np.random.poisson(lambdaP_now)
        t_now = t_now + dt
        lambdaP_now = N_now*beta*dt
        t.append(t_now)
        N_t.append(N_now)
        N_mean[i] = N_mean[i] + N_now/iterations
        i = i+1
        if i > i_max:
            i_max = i
        
        
    plt.plot(t, N_t, 'C1', alpha = 0.3 )

t_mean = t_mean[0:i_max]
N_mean = N_mean[0:i_max]
plt.plot(t_mean, N_mean, label = 'poveprečje')

t_mean = np.array(t_mean)
N_exact = N*np.exp(-beta*t_mean)

plt.plot(t_mean, N_exact, 'C2' ,label = 'točna rešitev')
plt.ylabel('$N(t)$')
plt.xlabel('$t$')
if doLog:
    plt.yscale('log')
    plt.ylim(1,250)
plt.title('Smrti v populaciji, $N_0 = $ ' + str(N) + ', $\u03B2 =$ ' + str(beta) + ', $dt =$ ' + str(dt))
plt.legend()
plt.show()
