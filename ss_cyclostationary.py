'''

NEYMANN-PEARSON SIGNAL DETECTOR FOR
SPECTRUM SAMPLING IN COGNITIVE RADIO
BASED ON CYCLOSTATIONARITY

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in, kamath-abhijith.github.io

'''

# %% LOAD LIBRARIES

import os
import numpy as np

from tqdm import tqdm

from scipy.stats import chi2

from matplotlib import style
from matplotlib import rcParams
from matplotlib import pyplot as plt

import utils

# %% PLOT SETTINGS

plt.style.use(['science','ieee'])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cm"],
    "mathtext.fontset": "cm",
    "font.size": 24})

# %% PARAMETERS

Nd = 32
Nc = 8
K = 50
N = (K+1)*(Nc+Nd)
NUM_STATS = 1000

PFA = 0.05

SNR_MIN = -20
SNR_MAX = 6
SNR_STEP = 2

# %% DISTRIBUTION OF TEST STATISTIC

SNR = 100
noise_var = 1 / 10**(SNR/10)

stats_H0 = utils.cyclo_stat_H0(NUM_STATS, Nd, Nc, K, noise_var)
stats_H1 = utils.cyclo_stat_H1(NUM_STATS, Nd, Nc, K, noise_var)

# %% PLOTS :: DISTRIBUTION OF TEST STATISTIC

os.makedirs('./results/', exist_ok=True)
path = './results/'

plt.figure(figsize=(12,6))
ax = plt.gca()
utils.plot_gaussian(np.linspace(-2*1e-10,2*1e-10), 0, Nc/(2*K)*noise_var**2,
    ax=ax, show=False)
utils.plot_histogram(np.real(stats_H0), 256, ax=ax,
    xaxis_label=r'$\bar{T}(\mathbf{y})$',
    title_text=r'$T(\mathbf{y})$ under $\mathcal{H}_0$', show=True, save=path+'cycDist_H0R')

plt.figure(figsize=(12,6))
ax = plt.gca()
utils.plot_gaussian(np.linspace(-2*1e-10,2*1e-10), 0, Nc/(2*K)*noise_var**2,
    ax=ax, show=False)
utils.plot_histogram(np.imag(stats_H0), 256, ax=ax,
    xaxis_label=r'$\tilde{T}(\mathbf{y})$',
    title_text=r'$T(\mathbf{y})$ under $\mathcal{H}_0$', show=True, save=path+'cycDist_H0I')

plt.figure(figsize=(12,6))
ax = plt.gca()
utils.plot_gaussian(np.linspace(6,10), Nc, Nc/K*(1+noise_var+(noise_var**2)/2),
    ax=ax, show=False)
utils.plot_histogram(np.real(stats_H1), 256, ax=ax,
    xaxis_label=r'$\bar{T}(\mathbf{y})$', xlimits=[6,10],
    title_text=r'$T(\mathbf{y})$ under $\mathcal{H}_1$', show=True, save=path+'cycDist_H1R')

plt.figure(figsize=(12,6))
ax = plt.gca()
utils.plot_gaussian(np.linspace(-2*1e-5,2*1e-5), 0, Nc/K*(noise_var+(noise_var**2)/2),
    ax=ax, show=False)
utils.plot_histogram(np.imag(stats_H1), 256, ax=ax,
    xaxis_label=r'$\tilde{T}(\mathbf{y})$', xlimits=[-2*1e-5,2*1e-5],
    title_text=r'$T(\mathbf{y})$ under $\mathcal{H}_1$', show=True, save=path+'cycDist_H1I')

# %% MONTE CARLO SIMULATIONS // CLEAN PARAMETERS

SNRS = np.arange(SNR_MIN, SNR_MAX, SNR_STEP)

true_PFA = np.zeros(len(SNRS))
est_PFA = np.zeros(len(SNRS))
est_PD = np.zeros(len(SNRS))

for itr, SNR in tqdm(enumerate(SNRS)):
    noise_var = 1 / 10**(SNR/10)
    threshold = chi2.isf(q=PFA, df=2) * Nc / K * (noise_var**2)

    stats_H0 = utils.cyclo_stat_H0(NUM_STATS, Nd, Nc, K, noise_var)
    stats_H1 = utils.cyclo_stat_H1(NUM_STATS, Nd, Nc, K, noise_var)

    false_alarms = sum(np.square(np.abs(stats_H0)) > threshold)
    detections = sum(np.square(np.abs(stats_H1)) > threshold)

    est_PFA[itr] = false_alarms / NUM_STATS
    est_PD[itr] = detections / NUM_STATS

    true_PFA[itr] = PFA

# %% PLOTS :: MONTE CARLO SIMULATIONS // CLEAN PARAMETERS

plt.figure(figsize=(12,6))
ax = plt.gca()
utils.plot_signal(SNRS, est_PD, ax=ax, plot_colour='blue',
    legend_label=r'ESTIMATED $P_{D}$', yaxis_label=r'$P_{D}$',
    xaxis_label=r'$\mathrm{SNR}$', show=True, save=path+'cycProb_PD')

plt.figure(figsize=(12,6))
ax = plt.gca()
utils.plot_signal(SNRS, true_PFA, ax=ax, plot_colour='green',
    legend_label=r'TRUE $P_{FA}$', show=False)
utils.plot_signal(SNRS, est_PFA, ax=ax, plot_colour='blue',
    legend_label=r'ESTIMATED $P_{FA}$', yaxis_label=r'$P_{FA}$',
    xaxis_label=r'$\mathrm{SNR}$', ylimits=[0,2*PFA], show=True,
    save=path+'cycProb_PFA')

# %% MONTE CARLO SIMULATIONS // NOISY PARAMETERS

SNRS = np.arange(SNR_MIN, SNR_MAX, SNR_STEP)

true_PFA = np.zeros(len(SNRS))
est_PFA = np.zeros(len(SNRS))
est_PD = np.zeros(len(SNRS))

for itr, SNR in tqdm(enumerate(SNRS)):
    noise_var = 1 / 10**(SNR/10)
    noise_var = noise_var * 10**(1/10)
    threshold = chi2.isf(q=PFA, df=2) * Nc / K * (noise_var**2)

    stats_H0 = utils.cyclo_stat_H0(NUM_STATS, Nd, Nc, K, noise_var)
    stats_H1 = utils.cyclo_stat_H1(NUM_STATS, Nd, Nc, K, noise_var)

    false_alarms = sum(np.square(np.abs(stats_H0)) > threshold)
    detections = sum(np.square(np.abs(stats_H1)) > threshold)

    est_PFA[itr] = false_alarms / NUM_STATS
    est_PD[itr] = detections / NUM_STATS

    true_PFA[itr] = PFA

# %% PLOTS :: MONTE CARLO SIMULATIONS // NOISY PARAMETERS

plt.figure(figsize=(12,6))
ax = plt.gca()
utils.plot_signal(SNRS, est_PD, ax=ax, plot_colour='blue',
    legend_label=r'ESTIMATED $P_{D}$', yaxis_label=r'$P_{D}$',
    xaxis_label=r'$\mathrm{SNR}$', show=True, save=path+'cycProb_PD_Noisy')

plt.figure(figsize=(12,6))
ax = plt.gca()
utils.plot_signal(SNRS, true_PFA, ax=ax, plot_colour='green',
    legend_label=r'TRUE $P_{FA}$', show=False)
utils.plot_signal(SNRS, est_PFA, ax=ax, plot_colour='blue',
    legend_label=r'ESTIMATED $P_{FA}$', yaxis_label=r'$P_{FA}$',
    xaxis_label=r'$\mathrm{SNR}$', ylimits=[0,2*PFA], show=True,
    save=path+'cycProb_PFA_Noisy')
