import random as rnd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from scipy.sparse import diags

matplotlib.style.use('seaborn')
PlotT = 1

Z0 = 200
L0 = 50
ratio = 10
alpha = 1
beta = alpha/ratio
dt = 0.001
iterations = 10

L_t, Z_t, t_t = [], [], []
L_t.append(L0)
Z_t.append(Z0)
L = L0
Z = Z0
t_now = 0
t_t.append(t_now) 
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
    L_t.append(L_new)
    Z_t.append(Z_new)
    t_t.append(t_now)
    L = L_new
    Z = Z_new

t_death = t_now

plt.plot(Z_t, L_t)
plt.title('Zajci in lisice, $\u03B1/\u03B2 = $' + str(ratio) + ', $Z_0 = $' + str(Z0) + ', $L_0 = $' + str(L0) +  '$, dt = $' + str(dt))
plt.xlabel('# zajcev')
plt.ylabel('# lisic')
plt.scatter(Z0, L0, c = 'red')
plt.scatter(Z_t[-1], L_t[-1], c = 'green')
plt.show()
if PlotT:
    plt.plot(t_t, Z_t, label = 'Zajci')
    plt.plot(t_t, L_t, label = 'Lisice')
    plt.xlabel('t')
    plt.ylabel('N')
    plt.legend()
    plt.annotate('$t_{death} = $' + str(round(t_death,2)) , xy=(0.65, 0.90), xycoords='axes fraction', backgroundcolor = 'white', fontsize = 'large')
    plt.title('Zajci in lisice, $\u03B1/\u03B2 = $' + str(ratio) + ', $Z_0 = $' + str(Z0) + ', $L_0 = $' + str(L0) +  '$, dt = $' + str(dt))
    plt.show()