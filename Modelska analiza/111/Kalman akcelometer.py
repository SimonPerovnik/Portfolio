import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.style.use('seaborn')

PlotData = 0

#Read and prepare data

#Measurments

list = []
file = open(r'E:\Faks\4.letnik\Modelska_analiza_I\111\kalman_relative_data.dat', "r")
list = file.readlines()
file.close()

t, x, y, at, ar = [],[],[],[],[]


for lines in list:
    lines = lines.split(' ')
    t.append(float(lines[0]))
    x.append(float(lines[1]))
    y.append(float(lines[2]))
    at.append(float(lines[3]))
    ar.append(float(lines[4][:-1]))
t = np.array(t)
x = np.array(x)
y = np.array(y)
at = np.array(at)
ar = np.array(ar)

#Control

list = []
file = open(r'E:\Faks\4.letnik\Modelska_analiza_I\111\kalman_cartesian_kontrola.dat', "r")
list = file.readlines()
file.close()

t_k, x_k, y_k, vx_k, vy_k, = [],[],[],[],[]


for lines in list:
    lines = lines.split(' ')
    t_k.append(float(lines[0]))
    x_k.append(float(lines[1]))
    y_k.append(float(lines[2]))
    vx_k.append(float(lines[3]))
    vy_k.append(float(lines[4]))
t_k = np.array(t_k)
x_k = np.array(x_k)
y_k = np.array(y_k)
vx_k = np.array(vx_k)
vy_k = np.array(vy_k)

if PlotData:
    gs = gridspec.GridSpec(3, 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0, :]) # row 0, across all col
    ax2 = fig.add_subplot(gs[1, :]) # row 1, across all col
    ax3 = fig.add_subplot(gs[2, :]) # row 2, across all col
    ax1.set_ylabel('$y(x)$', rotation = 0)
    ax1.set_xlabel('$x$')
    ax2.set_ylabel('$v_x(t), v_y(t)$', rotation = 0)
    ax2.set_xlabel('$t$')
    ax3.set_ylabel('$a_x(t), a_y(t)$', rotation = 0)
    ax3.set_xlabel('$t$')
    ax1.plot(x,y)
    ax2.plot(t,vx, label = '$v_x$', lw=0.7)
    ax2.plot(t,vy, label = '$v_y$', lw=0.7)
    #ax2.plot(t,np.sqrt(vx**2+vy**2), label = '$a_y$', lw=0.8)
    ax3.plot(t,at, label = '$a_x$', lw=0.7)
    ax3.plot(t,ar, label = '$a_y$', lw=0.7)
    #ax3.plot(t,np.sqrt(ax**2+ay**2), label = '$a_y$', lw=0.8)
    ax2.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1)
    ax3.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1)
    plt.tight_layout()
    plt.show()

#Prepare stuff for Kalman filter

def kalmanF():
    F = np.block([
                 [np.eye(2), np.eye(2)*deltaT],
                 [np.eye(2)*0, np.eye(2)]])
    return F
def c(n, x_n):
    x, y, vx, vy = x_n
    u = np.array([0, 0, at[n] * deltaT, ar[n] * deltaT ]).T
    B_n_vv = 1/(np.linalg.norm([vx, vy])) * np.array([ [vx, -vy], [vy, vx] ])
    B_n = np.block([
                   [np.eye(2), np.zeros((2,2))],
                   [np.zeros((2,2)), B_n_vv]
                   ])
    return B_n @ u
def kalmanQ(n, x_n, P_n):
    P_n_vv = P_n[2:,2:]
    x, y, vx, vy = x_n
    v_n = np.array([vx, vy]).T
    a_n = np.array([at[n], ar[n]]).T
    R = np.array([ [0,-1], [1,0]] )
    Q_old = np.diag([0, 0, sigma_a**2 * deltaT**2, sigma_a**2 * deltaT**2])
    B_n_vv = 1/(np.linalg.norm([vx, vy])) * np.array([ [vx, -vy], [vy, vx] ])
    Q_n = (R @ v_n @ P_n_vv @ R @ v_n)*1/(np.linalg.norm(v_n)**4) * np.tensordot(B_n_vv @ R @ a_n, B_n_vv @ R @ a_n, axes = 0) * deltaT**2
    Q_new = np.block([
                    [np.zeros((2,2)), np.zeros((2,2))],
                    [np.zeros((2,2)), Q_n]    
                    ])
    return Q_old + Q_new
def kalmanR(n):
    return np.diag([sigma_xy**2, sigma_xy**2, 0.5, 0.5])
def sigma_v(n):
    return max(0.2778,np.sqrt(vx[n]**2 + vy[n]**2) )
def kalmanH():
    return np.diag([1,1,0,0])

sigma_a = 0.05
sigma_xy = 25
deltaT = t[1]-t[0]
N = len(t)

z = np.array([x,y, x, y]).T
x_0 = np.array([x[0], y[0],2.872456927535406, -7.260569688219962])
x_n = x_0
P_0 = np.eye(4)*5
P_n = P_0
F = kalmanF()
H = kalmanH()
Q = kalmanQ(0, x_n, P_n)
R = kalmanR(0)
x_n_list = [x_0]
P_n_list = [P_0]
K_list = [0]
residual_list = [np.linalg.norm(z[0] - H @ x_0)]
for n in range(1,N):
    R = kalmanR(n)
    x_napoved = F @ x_n + c(n, x_n)
    P_napoved = F @ P_n @ F.T + kalmanQ(n, x_n, P_n)

    K = P_napoved @ H.T @ np.linalg.inv( H @ P_napoved @ H.T + R )
    x_n = x_napoved + K @ (z[n] - H @ x_napoved)
    P_n = (np.eye(len(x_0)) - K @ H) @ P_napoved
    x_n_list.append(x_n)
    P_n_list.append(P_n)
    K_list.append(K)
    residual_list.append(np.linalg.norm(z[n] - H @ x_n))

x_kalman = np.transpose(x_n_list)[0]
y_kalman = np.transpose(x_n_list)[1]
vx_kalman = np.transpose(x_n_list)[2]
vy_kalman = np.transpose(x_n_list)[3]

#Plot s(t)
plt.plot(x_kalman, y_kalman, label = 'kalman')
plt.plot(x_k, y_k, label = 'kontrola')
plt.plot(x,y, label= 'meritve', alpha=0.4)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1)
plt.show()

#Plot s(t)
plt.plot(x_kalman, y_kalman, label = 'kalman')
plt.plot(x_k, y_k, label = 'kontrola')
plt.plot(x,y, label= 'meritve', alpha=0.4, marker='o')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1)
plt.show()

#Plot vx(t), yv(t)
plt.plot(t, vx_kalman)
plt.plot(t, vy_kalman)
#plt.show()

#Plot residuals
plt.clf()
plt.yscale('log')
plt.ylabel('$r_i$')
plt.xlabel('$t$')
plt.plot(t, residual_list)
plt.title('$ \\langle r_i \\rangle = $' + str(round(np.mean(residual_list), 2)))
plt.plot([0,t[-1]], [np.mean(residual_list), np.mean(residual_list)])
plt.show()

#Plot variances and covariance
P_n_list_nonzero = np.transpose([ [i[0][0], i[2][2],i[0][2]] for i in P_n_list])
plt.plot(t, P_n_list_nonzero[0], label='$\sigma_{x,y}^2$')
plt.plot(t, P_n_list_nonzero[1], label='$\sigma_{v_x,v_y}^2$')
plt.plot(t, P_n_list_nonzero[2], label='$\sigma_{x} \sigma_{v_x}$')
plt.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1)
plt.xlabel('$t$')
plt.yscale('log')
plt.title('$P_0 = diag(5,5,5,5)$')
plt.show()


# plt.plot(t, np.abs( np.sqrt(x**2+y**2) - np.sqrt(x_kalman**2+y_kalman**2) ) )
# plt.show()

plt.scatter(x_k - x, y_k - y, label='Meritve', marker= 'x')
plt.scatter(x_k - x_kalman, y_k - y_kalman, label='Kalman', marker= 'x')
plt.xlabel('$x - x_k$')
plt.ylabel('$y - y_k$', rotation= 0)
plt.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1)
plt.show()


# plt.scatter(vx_k - vx_kalman, vy_k - vy_kalman, label='Kalman', marker= 'x')
# plt.scatter(vx_k - vx, vy_k - vy, label='Meritve', marker= 'x')
# plt.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1)
# plt.show()

# Plot comparisson between measurments and Kalman
# fig, (ax1, ax2) = plt.subplots(1, 2)

# ax1.scatter(x_k - x, y_k - y, label='Meritve', marker= 'x')
# ax1.scatter(x_k - x_kalman, y_k - y_kalman, label='Kalman', marker= 'x')
# ax1.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1)
# ax1.set_xlabel('$x - x_k$')
# ax1.set_ylabel('$y - y_k$', rotation= 0)
# ax1.axis('equal')

# ax2.scatter(vx_k - vx_kalman, vy_k - vy_kalman, label='Kalman', marker= 'x')
# ax2.scatter(vx_k - vx, vy_k - vy, label='Meritve', marker= 'x')
# ax2.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1)
# ax2.set_xlabel('$v_x - v_{x_k}$')
# ax2.set_ylabel('$v_y - v_{y_k}$', rotation= 0)
# ax2.axis('equal')
# plt.suptitle('Primerjava med meritvami in filtriranimi podatki')
# plt.show()
