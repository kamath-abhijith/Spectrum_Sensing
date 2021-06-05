'''

NEYMANN-PEARSON SIGNAL DETECTOR FOR
SPECTRUM SAMPLING IN COGNITIVE RADIO
BASED ON SIGNAL ENERGY

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

# %% MONTE CARLO SIMULATIONS // CLEAN PARAMETERS

SNRS = np.arange(SNR_MIN, SNR_MAX, SNR_STEP)

true_PFA = np.zeros(len(SNRS))
true_PD = np.zeros(len(SNRS))
est_PFA = np.zeros(len(SNRS))
est_PD = np.zeros(len(SNRS))

for itr, SNR in tqdm(enumerate(SNRS)):
    noise_var = 1 / 10**(SNR/10)
    threshold = chi2.isf(q=PFA, df=N) * noise_var

    stats_H0 = utils.energy_stat_H0(NUM_STATS, Nd, Nc, K, noise_var)
    stats_H1 = utils.energy_stat_H1(NUM_STATS, Nd, Nc, K, noise_var)

    false_alarms = sum(stats_H0 > threshold)
    detections = sum(stats_H1 > threshold)

    est_PFA[itr] = false_alarms / NUM_STATS
    est_PD[itr] = detections / NUM_STATS

    true_PFA[itr] = PFA
    true_PD[itr] = chi2.sf(x=threshold / (1 + noise_var), df=N)

# %% PLOTS // CLEAN PARAMETERS

os.makedirs('./results/', exist_ok=True)
path = './results/'

plt.figure(figsize=(12,6))
ax = plt.gca()
utils.plot_signal(SNRS, true_PD, ax=ax, plot_colour='green',
    legend_label=r'TRUE $P_D$', show=False)
utils.plot_signal(SNRS, est_PD, ax=ax, plot_colour='blue',
    legend_label=r'ESTIMATED $P_{D}$', yaxis_label=r'$P_{D}$',
    xaxis_label=r'$\mathrm{SNR}$', show=True,
    save=path+'eneProb_PD')

plt.figure(figsize=(12,6))
ax = plt.gca()
utils.plot_signal(SNRS, true_PFA, ax=ax, plot_colour='green',
    legend_label=r'TRUE $P_{FA}$', show=False)
utils.plot_signal(SNRS, est_PFA, ax=ax, plot_colour='blue',
    legend_label=r'ESTIMATED $P_{FA}$', yaxis_label=r'$P_{FA}$',
    xaxis_label=r'$\mathrm{SNR}$', ylimits=[0,2*PFA], show=True,
    save=path+'eneProb_PFA')

# %% MONTE CARLO SIMULATIONS // NOISY PARAMETERS

SNRS = np.arange(SNR_MIN, SNR_MAX, SNR_STEP)

true_PFA = np.zeros(len(SNRS))
true_PD = np.zeros(len(SNRS))
est_PFA = np.zeros(len(SNRS))
est_PD = np.zeros(len(SNRS))

for itr, SNR in tqdm(enumerate(SNRS)):
    noise_var = 1 / 10**(SNR/10)
    noise_var = noise_var * 10**(1/10)
    threshold = chi2.isf(q=PFA, df=N) * noise_var

    stats_H0 = utils.energy_stat_H0(NUM_STATS, Nd, Nc, K, noise_var)
    stats_H1 = utils.energy_stat_H1(NUM_STATS, Nd, Nc, K, noise_var)

    false_alarms = sum(stats_H0 > threshold)
    detections = sum(stats_H1 > threshold)

    est_PFA[itr] = false_alarms / NUM_STATS
    est_PD[itr] = detections / NUM_STATS

    true_PFA[itr] = PFA
    true_PD[itr] = chi2.sf(x=threshold / (1 + noise_var), df=N)

# %% PLOTS // NOISY PARAMETERS

os.makedirs('./results/', exist_ok=True)
path = './results/'

plt.figure(figsize=(12,6))
ax = plt.gca()
utils.plot_signal(SNRS, true_PD, ax=ax, plot_colour='green',
    legend_label=r'TRUE $P_D$', show=False)
utils.plot_signal(SNRS, est_PD, ax=ax, plot_colour='blue',
    legend_label=r'ESTIMATED $P_{D}$', yaxis_label=r'$P_{D}$',
    xaxis_label=r'$\mathrm{SNR}$', show=True,
    save=path+'eneProb_PD_Noisy')

plt.figure(figsize=(12,6))
ax = plt.gca()
utils.plot_signal(SNRS, true_PFA, ax=ax, plot_colour='green',
    legend_label=r'TRUE $P_{FA}$', show=False)
utils.plot_signal(SNRS, est_PFA, ax=ax, plot_colour='blue',
    legend_label=r'ESTIMATED $P_{FA}$', yaxis_label=r'$P_{FA}$',
    xaxis_label=r'$\mathrm{SNR}$', ylimits=[0,2*PFA], show=True,
    save=path+'eneProb_PFA_Noisy')

# %% THRESHOLD COMPARISONS

prior1 = 0.2
prior0 = 1-prior1
SNRS = np.arange(SNR_MIN, SNR_MAX, SNR_STEP)

threshold_NP = np.zeros(len(SNRS))
threshold_BD = np.zeros(len(SNRS))
for itr, SNR in tqdm(enumerate(SNRS)):
    noise_var = 1 / 10**(SNR/10)
    noise_var = noise_var * 10**(1/10)
    
    threshold_NP[itr] = chi2.isf(q=PFA, df=N) * noise_var
    threshold_BD[itr] = 2 * (1+noise_var) * noise_var * \
        (N/2 * np.log((1+noise_var)/noise_var) + np.log(prior0/prior1))

# %% PLOTS :: THRESHOLD COMPARISON

os.makedirs('./results/', exist_ok=True)
path = './results/'

plt.figure(figsize=(12,6))
ax = plt.gca()
utils.plot_signal(SNRS, threshold_NP, ax=ax,
    legend_label=r'Neymann-Pearson Detector', show=False)
utils.plot_signal(SNRS, threshold_BD, ax=ax, plot_colour='green',
    ylimits=[0,0.5*1e6], legend_label=r'Bayes Detector', show=True,
    save=path+'thresholds')
# %%
