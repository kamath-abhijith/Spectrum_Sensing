'''

TOOLS FOR SPECTRUM SENSING FOR COGNITIVE RADIO

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in, kamath-abhijith.github.io

'''

# %% LOAD LIBRARIES

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from scipy.stats import multivariate_normal

# %% PLOTTING FUNCTIONS

def plot_signal(x, y, ax=None, plot_colour='blue', xaxis_label=None,
    yaxis_label=None, title_text=None, legend_label=None, legend_show=True,
    legend_loc='upper right', line_style='-', line_width=None,
    show=False, xlimits=[-20,6], ylimits=[0,1], save=None):
    '''
    Plots signal with abscissa in x and ordinates in y 

    '''
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    plt.plot(x, y, linestyle=line_style, linewidth=line_width, color=plot_colour,
        label=legend_label)
    if legend_label and legend_show:
        plt.legend(loc=legend_loc, frameon=True, framealpha=0.8, facecolor='white')
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)

    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.title(title_text)

    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0E}'))

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return

def plot_histogram(x, num_bins, ax=None, plot_colour='blue', xaxis_label=None,
    yaxis_label=None, title_text=None, xlimits=[-2*1e-10,2*1e-10], show=False, save=None):
    '''
    Plots histogram of data in x

    '''

    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    plt.hist(x, bins=num_bins, color=plot_colour, density=True)

    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title_text)

    plt.xlim(xlimits)
    # plt.ylim([0,30])

    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0E}'))

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return

def plot_gaussian(xvals, mean, var, ax=None, plot_colour='red',
    xaxis_label=None, yaxis_label=None, line_style='-', line_width=2,
    title_text=None, show=False, save=None):
    '''
    Plots 1D Gaussian with mean and variance

    '''

    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    gaussian = multivariate_normal.pdf(xvals, mean, var)
    plt.plot(xvals, gaussian, linestyle=line_style, linewidth=line_width,
        color=plot_colour)

    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title_text)

    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0E}'))

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return

# %% SIGNALS

def generate_QPSK_block(Nd, Nc):
    '''
    Generates a QPSK block with unit variance

    :param Nd: IFFT length
    :param Nc: length of cyclic repetition

    :return: OFDM signal block

    '''
    
    s = np.zeros(Nd)
    for k in range(Nd):
        s[k] = (2*np.random.binomial(1, 0.5)-1)/np.sqrt(2) + \
            1j * (2*np.random.binomial(1, 0.5)-1)/np.sqrt(2)

    x = np.fft.ifft(s) * np.sqrt(2*Nd)

    return np.hstack([x[:Nc], x])

def generate_QPSK_signal(Nd, Nc, K):
    '''
    Generate a QPSK signal with unit variance

    :param Nd: IFFT length of each block
    :param Nc: length of cyclic repetition
    :param K: number of OFDM blocks

    :return: K+1 OFDM signal blocks

    '''

    signal = generate_QPSK_block(Nd, Nc)
    for _ in range(K):
        signal = np.hstack([signal, generate_QPSK_block(Nd, Nc)])

    return signal

# %% DETECTORS

def energy_stat_H0(num_stats, Nd, Nc, K, noise_var):
    '''
    Generates signal energy statsitic for noise only hypothesis

    :param num_stats: number of realisations
    :param Nd: length of IFFT
    :param Nc: length of cyclic repetitions
    :param K: number of OFDM blocks
    :param noise_var: variance of AWGN channel

    :return: energy statistics

    '''
    
    stats = np.zeros(num_stats)
    for itr in range(num_stats):
        noise = np.sqrt(noise_var)*np.random.randn((K+1)*(Nd+Nc))

        stats[itr] = np.sum(np.square(np.abs(noise)))

    return stats

def energy_stat_H1(num_stats, Nd, Nc, K, noise_var):
    '''
    Generates signal energy statsitic for signal present hypothesis

    :param num_stats: number of realisations
    :param Nd: length of IFFT
    :param Nc: length of cyclic repetitions
    :param K: number of OFDM blocks
    :param noise_var: variance of AWGN channel

    :return: energy statistics

    '''
    
    stats = np.zeros(num_stats)
    for itr in range(num_stats):
        signal = generate_QPSK_signal(Nd, Nc, K)
        noise = np.sqrt(noise_var)*np.random.randn((K+1)*(Nd+Nc))

        stats[itr] = np.sum(np.square(np.abs(signal + noise)))

    return stats

def cyclo_stat_H0(num_stats, Nd, Nc, K, noise_var):
    '''
    Generates cyclostationary ACF statistic for noise only hypothesis

    :param num_stats: number of realisations
    :param Nd: length of IFFT
    :param Nc: length of cyclic repetitions
    :param K: number of OFDM blocks
    :param noise_var: variance of AWGN channel

    :return: cyclostationary ACF  

    '''
    
    stats = np.zeros(num_stats, dtype=np.complex)
    for itr in range(num_stats):
        noise = np.sqrt(noise_var)*np.random.randn((K+1)*(Nd+Nc),2).view(np.complex)
        y = noise

        stat = np.complex(0)
        for n in range(Nc):
            for k in range(K):
                stat += y[n+k*(Nc+Nd)] * np.conjugate(y[n+k*(Nc+Nd)+Nd])

        stats[itr] = stat/K

    return stats

def cyclo_stat_H1(num_stats, Nd, Nc, K, noise_var):
    '''
    Generates cyclostationary ACF statistic for signal present hypothesis

    :param num_stats: number of realisations
    :param Nd: length of IFFT
    :param Nc: length of cyclic repetitions
    :param K: number of OFDM blocks
    :param noise_var: variance of AWGN channel

    :return: cyclostationary ACF  

    '''
    
    stats = np.zeros(num_stats, dtype=np.complex)
    for itr in range(num_stats):
        signal = generate_QPSK_signal(Nd, Nc, K)[:,None]
        for k in range(K):
            signal[k*(Nc+Nd):k*(Nc+Nd)+Nc] = signal[k*(Nc+Nd)+Nd:(k+1)*(Nc+Nd)]
        noise = np.sqrt(noise_var)*np.random.randn((K+1)*(Nd+Nc),2).view(np.complex)
        y = signal + noise

        stat = np.complex(0)
        for n in range(Nc):
            for k in range(K):
                stat += y[n+k*(Nc+Nd)] * np.conjugate(y[n+k*(Nc+Nd)+Nd])

        stats[itr] = stat/K

    return stats