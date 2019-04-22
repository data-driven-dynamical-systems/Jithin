import numpy as np
import matplotlib.pyplot as plt

num_sims = 500 ### display five runs


K = 6435*np.pi/16384

sigma = 1

def mu_func(y, t): 
    """the mu term""" 
    return (np.cos(y)**16)/K - 1/(2*np.pi)

def sigma_func(y, t): 
    """the sigma. term""" 
    return sigma
    
def dW(delta_t): 
    """Weiner jump"""
    return np.random.normal(size = num_sims, scale = np.sqrt(delta_t), loc =0.0)



def stochastic_euler(mu_func,sigma_func,t_init,t_end,y_init,N,num_sims):
    dt     = float(t_end - t_init) / N
    ts    = np.arange(t_init, t_end, dt)
    ys    = np.zeros((N,num_sims))

    ys[0] = y_init
    for i in range(1, N):
            t = (i-1) * dt
            y = ys[i-1]
            ys[i,:] = ys[i-1,:] + mu_func(y, t) * dt +  dW(dt)
    return ys,ts


