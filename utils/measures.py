# import data_generate as data_gen
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats
from mpl_toolkits.mplot3d import Axes3D 
from statistics import mean, stdev
import math
import sys
import seaborn as sns
from utils.dynamical_systems import generate_points

def invariant_measure(phi, n_points, t_start, t_end, y0, sd, distribution = "uniform", seed = None):
    """
    phi : dynamical system, supports batch input
    """
    n_dim = y0.shape[0]
    init_conds = generate_points(n_points, y0, sd, distribution=distribution, seed = seed)
    measure = phi.integrate(init_conds, t_end)[:, t_start: , :] # (batch, t_end-t_start, n_dim)
    
    return measure.reshape(-1, n_dim)  # (n_points * (t_end - t_start), n_dim)

# def pushforward(phi, measure, t=1):

def plot_measure(measure, axes = (0,1), num_bins = None, plot_type = 'kde'):
    if len(measure.shape) == 2:
        measure = measure.reshape((1,measure.shape[0], measure.shape[1]))
    
    for i in range(measure.shape[0]):
        x = measure[i,:,axes[0]]
        y = measure[i,:,axes[1]]
        if plot_type == 'kde':
            sns.kdeplot(x=x, y=y, shade = True)
        elif plot_type == 'hist':
            plt.hist2d(x,y, num_bins)
        else:
            print('Plot type not supported.')
        # plt.hexbin(x,y,gridsize = num_bins, bins = log )
        # could also use gaussian kdes, or try 3d plots

def sample_measure(measure, n_sample, seed = None):
    np.random.seed(seed)
    indices = np.random.choice(measure.shape[0], size=n_sample, replace = True)
    return np.array([measure[i] for i in indices])

# wasserstein_distance_nd(mu1, mu2)

def rbf(x,y,sigma):
    return np.exp( - np.linalg.norm(x-y)**2/(2*sigma**2))

def MMD(sample1,sample2,kernel):
    m = len(sample1)
    n = len(sample2)
    xx = 0
    for i in range(m):
        for j in range(m):
            if i != j:
                xx += kernel(sample1[i], sample1[j])
    xx = xx/(m*(m-1))
    print('xx',xx)
    
    yy=0
    for i in range(n):
        for j in range(n):
            if i != j:
                yy += kernel(sample2[i], sample2[j])
    yy = yy/(n*(n-1))
    print('yy',yy)

    xy=0
    for i in range(m):
        for j in range(n):
            xy += kernel(sample1[i], sample2[j])
    xy = xy/(m*n)

    print('xy', xy)

    return np.sqrt(xx + yy - 2*xy)
