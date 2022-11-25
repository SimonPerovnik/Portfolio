import random as rnd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from scipy.sparse import diags



matplotlib.style.use('seaborn')


beta = 1
beta_r = 4*beta
beta_s = 5*beta
N = 25
interval = (N*3)
N_ext = N + interval
dt = 0.0001

accuracy = 0.57
numTicks = 8

def TransitionMatrix (N_ext):
    N = N_ext
    diag1 = np.array([beta_r * n * dt for n in range(0, N)])
    diag2 = np.array([1 - (beta_r + beta_s)* n * dt for n in range(0, N+1)])
    diag3 = np.array([beta_s * n * dt for n in range(1, N+1)])
    diag3[-1] = N*5*dt
    diag2[-1] = 1-N*5*dt


    k = [diag1,diag2,diag3]
    offset = [-1,0,1]
    return diags(k,offset).toarray()


M = TransitionMatrix(N_ext)
x = np.concatenate((np.zeros(N), np.ones(1), np.zeros(interval)), axis=None)
t_now = 0
N_t = []
while (x[0] < accuracy):
    if t_now > 15: break 
    t_now = t_now + dt
    x = M.dot(x)
    N_t.append(x)

N_t = np.transpose(np.array(N_t))
t_death = t_now

c = plt.imshow(N_t, cmap = 'plasma', vmin = 0.0001, vmax = 1, aspect = 'auto',
               origin = 'lower', interpolation = 'nearest', norm=matplotlib.colors.LogNorm())
plt.ylim(0,40)
a_list = np.linspace(0, t_now, numTicks+1, endpoint=True)
plt.xticks(np.arange(0, int(t_death/dt), step = int(t_death/dt/numTicks)), labels= [round(num, 1) for num in a_list])
plt.colorbar(c)
plt.xlabel('$t$')
plt.ylabel('$N$')
plt.title('Čas izumrtja - $N = $' + str(N) + ', $\u03B2 =$ ' + str(beta) +' pri verjetnosti ' + str(accuracy*100)+'%')
plt.annotate('$t_{death} = $' + str(round(t_death,2)) , xy=(0.75, 0.90), xycoords='axes fraction', backgroundcolor = 'white', fontsize = 'large')
plt.show()


