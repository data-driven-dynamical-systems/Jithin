from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

N = 50 #number of oscillators
K = 0.5 #coupling

def Kuramoto(theta,t, omega, K):
    'derivatives in the kuramoto model'
    N = len(theta)
    dtheta = np.zeros(N)
    for i in range(N):
        dtheta[i] = omega[i] + (K/N)*np.sum(np.sin(theta-theta[i]))
    return dtheta
x1 = np.ones(N)
x0 = np.linspace(0,2*np.pi,N)
v0 = np.ones(N) # initial state (equilibrium)
x0[19] += 0.01 # add small perturbation to 20th variable
t = np.arange(0.0, 30.0, 0.01)

x = odeint(Kuramoto,x0,t, args =(v0,K))

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = plt.subplot(111, polar=True)
ax.grid(False)
ax.set_yticklabels([])
ax.set_xticklabels([])
pltvec = np.linspace(0,2999,20)
plt.ion()
for i in pltvec:
    i = int(i)
    plt.clf()
    ax = plt.subplot(111, polar=True)
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax.plot(x[i,:],x1,color='r', ls='none', marker='.')
    plt.pause(0.1)
    
plt.ioff()
plt.show()
