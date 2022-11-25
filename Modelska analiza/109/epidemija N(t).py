import random as rnd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from scipy.sparse import diags

matplotlib.style.use('seaborn')
PlotT = 1

N = 500
O0 = int(N*0.15)
I0 = int(N*0.01)
D0 = N - O0 - I0

alpha = 0.005
beta = 0.5
gamma = 0.003
dt = 0.01
iterations = 10

O_t, I_t, D_t, t_t = [], [], [], []
N_t = []
O_t.append(O0)
I_t.append(I0)
D_t.append(D0)
N_t.append(N)

O = O0
I = I0
D = D0
t_now = 0
t_t.append(t_now)

while( O > 0 ):
    t_now = t_now + dt
    if t_now > 1000:
        break
    D_new = (D  - np.random.poisson(alpha*D*O*dt) 
                + np.random.poisson(gamma*I*dt)) 
    O_new = (O  + np.random.poisson(alpha*D*O*dt) 
                - np.random.poisson(beta*O*dt))
    I_new = (I  + np.random.poisson(beta*O*dt) 
                - np.random.poisson(gamma*I*dt))
    D_t.append(D_new)
    O_t.append(O_new)
    I_t.append(I_new)
    N_t.append(D_new + O_new + I_new)
    t_t.append(t_now)
    D = D_new
    O = O_new
    I = I_new

t_death = t_now

plt.plot(t_t, N_t, label = 'Vsi')
plt.plot(t_t, D_t, label = 'Dovzetni')
plt.plot(t_t, O_t, label = 'Okuženi')
plt.plot(t_t, I_t, label = 'Imuni')

plt.title('Epidemija, $\u03B1 = $' + str(alpha) + ', \u03B2 = $' + str(beta) + ', \u03B3 = $' + str(gamma) 
          + ', $D_0 = $' + str(D0) + ', $O_0 = $' + str(O0) + ', $I_0 = $' + str(I0) +  '$, dt = $' + str(dt))
plt.xlabel('t')
plt.ylabel('N')
plt.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1)
plt.show()