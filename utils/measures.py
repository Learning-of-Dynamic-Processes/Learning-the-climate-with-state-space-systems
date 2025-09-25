#%%
# # import data_generate as data_gen
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

    mmd_2 = np.max(xx + yy - 2*xy, 0) # the above gives an unbiased estimate, but may be negative
    return np.sqrt(mmd_2)

def mmd_rbf_seq(x: np.ndarray, y: np.ndarray, sigma: float = 1.0, biased = False) -> np.ndarray:
    """
    Computes mmd with rbf kernel over a sequence of samples.
    Args:
        x: (n, T, d) array
        y: (m, T, d) array
        sigma: RBF bandwidth

    Returns:
        mmd: (T,) array of MMD values per time step
    """
    n, T, d = x.shape
    m = y.shape[0]

    x_norm = np.sum(x**2, axis = 2) # (n, T)
    y_norm = np.sum(y**2, axis = 2) # (m, T)

    Kxx = np.exp(
        -(
            x_norm[None, :, :] + x_norm[:, None, :] - 2 * np.einsum("ntd,mtd->nmt", x, x)
            ) / (2 * sigma**2)
    ) # (n, n, T)
    Kyy = np.exp(
        -(
            y_norm[None, :, :] + y_norm[:, None, :] - 2 * np.einsum("ntd,mtd->nmt", y, y)
            ) / (2 * sigma**2)
    ) # (m, m, T)
    Kxy = np.exp(
        -(
            x_norm[:, None, :] + y_norm[None, : , :] - 2* np.einsum("ntd,mtd->nmt", x, y)
        ) / (2 * sigma**2)
    ) # (n, m, T)
    
    if not(biased):
        for t in range(T):
            diag_id = np.arange(n)
            Kxx[diag_id, diag_id, :] = 0
            diag_id = np.arange(m)
            Kyy[diag_id, diag_id, :] = 0
        Kxx_mean = Kxx.sum(axis=(0,1)) / (n*(n-1))
        Kyy_mean = Kyy.sum(axis=(0,1)) / (m*(m-1))
    else:
        Kxx_mean = Kxx.sum(axis=(0,1)) / n**2
        Kyy_mean = Kyy.sum(axis=(0,1)) / m**2

    Kxy_mean = Kxy.sum(axis=(0,1)) / (n*m) # (T,)

    return Kxx_mean + Kyy_mean - 2 * Kxy_mean

def mmd_rbf_seq_lin_time(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Computes mmd with rbf kernel over a sequence of samples of equal size.
    Args:
        x: (n, T, d) array
        y: (n, T, d) array
        sigma: RBF bandwidth

    Returns:
        mmd: (T,) array of MMD values per time step
    """
    n, T, d = x.shape
    
    if y.shape[0] != n:
        assert f"samples x, y must have equal size, but have sizes {n} and {y.shape[0]}."
    
    n2 = np.floor(n/2)
    ids_1 = 2 * np.arange(n2).astype(int) 
    ids_2 = 2 * np.arange(n2).astype(int) + 1
    x1, x2 = x[ids_1], x[ids_2]   # (n2, T, d)
    y1, y2 = y[ids_1], y[ids_2]

    # Pairwise squared distances, fully vectorized
    Kxx = np.exp(-np.sum((x1 - x2) ** 2, axis=-1) / (2 * sigma**2))  # (n2, T)
    Kyy = np.exp(-np.sum((y1 - y2) ** 2, axis=-1) / (2 * sigma**2))
    Kxy12 = np.exp(-np.sum((x1 - y2) ** 2, axis=-1) / (2 * sigma**2))
    Kxy21 = np.exp(-np.sum((x2 - y1) ** 2, axis=-1) / (2 * sigma**2))

    Kxx_mean = Kxx.mean(axis=0)
    Kyy_mean = Kyy.mean(axis=0)
    Kxy12_mean = Kxy12.mean(axis=0)
    Kxy21_mean = Kxy21.mean(axis=0)
    

    return Kxx_mean + Kyy_mean - Kxy12_mean - Kxy21_mean

def mmd_bound(alpha, epsilon, m, n, K = 1, biased = False):
    """
    Finds bounds for a level alpha test that the MMD between
    two distributions is greater than epsilon.
    Args:
        alpha : level of test
        epsilon : testing distance between the two distributions
        m : sample 1 size
        n : sample 2 size
        K : 0 <= k(x,y) <= K for kernel k(,)
        biased : biased or unbiased test

    Returns:
        bound : if MMD_sample < bound, then, the true MMD
        between the two distributions is less than epsilon
        with probability 1 - alpha
    """
    if biased:
        delta_sq = 2 * K * (m+n) * (np.log(2) - np.log(alpha)) / (m*n) 

        bound = epsilon - 2 * (np.sqrt(K/m) + np.sqrt(K/n)) - np.sqrt(delta_sq)
    else:
        if m != n:
            assert f"unbiased test requires m=n but m is {m} and n is {n}"
        m2 = np.floor(m/2)
        delta_sq =  - 8 * K**2 * np.log(alpha) / m2 

        bound = epsilon - np.sqrt(delta_sq)

    return bound


#%%

def two_sample_test(m, alpha = 0.05, H0 = '===', epsilon_sq = None, biased = True, K = 1):
    """
    Test to differentiate between samples from two distributions.

    Params
        m : sample size (equal samples)
        alpha : p-value
        H0 : null hypothesis, options: '==', '>eps', '<eps'
        epsilon_sq : test value in case null hypothesis is not '=='
        biased : biased or unbiased statistic
        K : 0 <= k(x,y) <= K for the kernel k

    Returns
        squared critical value for acceptance/rejection
    """

    if biased:
        if H0 == '==':
            delta = 2 * np.sqrt((K/m) * ( - np.log(alpha)))
            z = np.sqrt( 2 * K / m) + delta
        else:
            epsilon = np.sqrt(epsilon_sq)
            delta = 2 * np.sqrt((K/m) * (- np.log(alpha/2)))
            if H0 == '>eps':
                z = epsilon - 4 * np.sqrt(K/m) - delta
            elif H0 == '<eps':
                z = epsilon + 4 * np.sqrt(K/m) + delta
            else:
                assert 'invalid null hypothesis H0'
        if z<0:
            raise Warning(f'cut off value z is negative z={z}')        
        z_sq = z**2

    else:
        t = 4 * K * np.sqrt( - np.log(alpha) / m)
        if H0 == '==':
            z_sq = t
        elif H0 == '>eps':
            z_sq = epsilon_sq - t
        elif H0 == '<eps':
            z_sq = epsilon_sq + t
        else:
            assert 'invalid null hypothesis H0'
        
    return z_sq



def sample_size_diff_two_sample_test(l_sq, h_sq, alpha = 0.05, biased = True, K = 1):
    """
    Sample size required to perform a two sample test
    to differentiate between samples from two distributions.

    Params
        l_sq, h_sq : low and high values (squared) to differentiate
        alpha : p-value
        biased : biased or unbiased statistic
        K : 0 <= k(x,y) <= K for the kernel k

    Returns
        sample size required to differentiate between high and low values
    """

    if biased:
        l, h = np.sqrt(l_sq), np.sqrt(h_sq)
        delta_1 = (h-l)/2
        m = 4 * K * (2 + np.sqrt( - np.log(alpha/2) ) )**2 / delta_1**2
    else:
        t = (h_sq - l_sq) / 2
        m = - 16 * K**2 * np.log(alpha) / t**2

    return np.ceil(m)

def sample_size_cut_off_two_sample_test(z_sq, alpha = 0.05,  H0 = '==', epsilon_sq = None, biased = True, K = 1):
    """
    Sample size required to perform a two sample test
    to obtain a given cut off statistic

    Params
        z_sq : squared cut off statistic
        alpha : p-value
        H0 : null hypothesis, options: '==', '>eps', '<eps'
        epsilon_sq : test value in case null hypothesis is not '=='
        biased : biased or unbiased statistic
        K : 0 <= k(x,y) <= K for the kernel k

    Returns
        sample size required to differentiate between high and low values
    """

    if biased:
        if H0 == '==':
            m = (2 * K / z_sq) * (1 + np.sqrt(-2 * np.log(alpha)))**2
        elif H0 == '>eps' or H0 == '<eps':
            epsilon = np.sqrt(epsilon_sq)
            z = np.sqrt(z_sq)
            m = 4 * K * ( 2 + np.sqrt( - np.log(alpha/2)))**2 / (epsilon - z)**2
        else:
            assert 'invalid null hypothesis H0'
        
    else:
        if H0 == '==':
            m = (4 * K / z_sq)**2 * (- np.log(alpha))
        elif H0 == '>eps' or H0 == '<eps':
            m = 16 * K**2 * (-np.log(alpha)) / (epsilon_sq - z_sq)**2
        else:
            assert 'invalid null hypothesis H0'
        
    return np.ceil(m)

def epsilon_sq_given_cut_off_two_sample_test(m, z_sq, alpha = 0.05,  H0 = '>eps', biased = True, K = 1):
    """
    epsilon_sq distinction given a cut off to perform a two sample test

    Params:
        m : sample size
        z_sq : cut off
        alpha : p-value
        H0 : null hypothesis, options: '>eps', '<eps'
        epsilon_sq : test value in case null hypothesis is not '=='
        biased : biased or unbiased statistic
        K : 0 <= k(x,y) <= K for the kernel k

    Returns
        epsilon_sq value
    """

    if biased:
        z = np.sqrt(z_sq)
        delta = 2 * np.sqrt(K/m) * ( 2 + np.sqrt( - np.log(alpha/2)))
        if H0 == '>eps':
            epsilon = z + delta
        elif H0 == '<eps':
            epsilon = z - delta
        else:
            assert 'invalid null hypothesis H0'
        epsilon_sq = epsilon**2
    else:
        t = 4 * K * np.sqrt(- np.log(alpha) / m)
        if H0 == '>eps':
            epsilon_sq = z_sq + t
        elif H0 == '<eps':
            epsilon_sq = z_sq - t
        else:
            assert 'invalid null hypothesis H0'
        
    return epsilon_sq
6
# #%%
# import matplotlib.pyplot as plt

# alpha = 0.05
# epsilon = 0
# K=1
# biased = False
# bounds = []
# for m in range(1, 1000):
#     bounds.append(0.002 - mmd_bound(alpha, epsilon, m, m, K, biased))

# plt.plot(bounds)
# plt.ylim(-1,1)
# plt.axhline(0)
# plt.show()
    
# # %%

# for i in range(1,1000):
#     if bounds[i] <0.25:
#         break
# print(i)


# %%
