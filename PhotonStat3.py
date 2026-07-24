# -*- coding: utf-8 -*-
"""
Lotte Gritter
l.gritter@uu.nl
"""
import cv2
import os
import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
from PIL import Image
import time
import math
from sklearn.linear_model import LinearRegression
#%% Select directory and set the number of frames used
dir = "C:/Users/lotte/OneDrive/Documenten/Universiteit/BSc/Scriptie/PoissonRuis/"
Nframes = 1000

# function that will give the counts for pixel (i,j) with shutter time s
def noise(i,j,s):
    k=1
    noise = []
    while k <Nframes:
        k=str(k)
        k= k.rjust(4, "0")
        if s==160:
            data = Image.open(os.path.join(dir,"PoissonNoise160/Basler acA2440-35um (24508848)_20260507_135518144_{}.tiff".format(k)))
        elif s==100:
            data = Image.open(os.path.join(dir,"PoissonNoise100/Basler acA2440-35um (24508848)_20260507_135754075_{}.tiff".format(k)))
        elif s==200:
            data = Image.open(os.path.join(dir,"PoissonNoise200/Basler acA2440-35um (24508848)_20260507_135650017_{}.tiff".format(k)))
        elif s==250:
            data = Image.open(os.path.join(dir,"PoissonNoise250/Basler acA2440-35um (24508848)_20260507_140023245_{}.tiff".format(k)))
        else:
            print(f"{s} is not a possible value for s.")
            break
        data = np.array(data)
        noise.append(data[i][j])
        # print(data_flat[i][j])
        k= int(k)+1
    noise = np.array(noise)    
    return noise

# print(noise)

# testim = Image.open(os.path.join(dir,"Basler acA2440-35um (24508848)_20260507_135518144_0001.tiff"))

#%% Choose a pixel and exposure time
counts = noise(268,325,250)

#%% Plot the histogram for the pixel and fit a Gaussian to it
# determine the mean and variance of the counts
mu = np.mean(counts)
std = np.sqrt(np.var(counts))

# fit a Gaussian to the data
x = np.arange(min(counts), max(counts) + 1)
pmf = stats.norm.pdf(x, mu, std)

bins = np.arange(min(counts), max(counts) + 2) - 0.5  # width-1 bins centered on integers

plt.figure(dpi=150)
plt.hist(counts, bins=bins, density=True, alpha=0.6, color='g', label='Camera Data')
plt.plot(x, pmf, 'ko-', linewidth=2, markersize=4, label=f'Gaussian approximation: μ={mu:.2f}, σ={std:.2f}')
plt.title("Poisson noise on camera")
plt.xlabel("Counts on camera pixel")
plt.ylabel("Probability")
plt.axis((230, 246, 0, 0.225))
plt.legend()
plt.show()


#%% Find the Fano factor for noise(i,j,s)
mean_counts = np.mean(counts)
var_counts = np.var(counts, ddof=1)  

fano_factor = var_counts / mean_counts  # should be ≈1 for true Poisson statistics

print(f"Sample mean:     {mean_counts:.3f}")
print(f"Sample variance: {var_counts:.3f}")
print(f"Fano factor (var/mean): {fano_factor:.3f}")

#%% Plot the mean and variance with errors (SEM) and error of the variance
mean = []
variance = []
mean_err = []
var_err = []
F = []
for i in [100,160,200,250]:
    counts = noise(278,325,i)
    mean.append(np.mean(counts))
    variance.append(np.var(counts))
    mean_err.append(np.std(counts)/np.sqrt(Nframes)) #SEM
    var_err.append(np.var(counts)*np.sqrt(2/(Nframes-1))) #error for variance
    F.append(np.var(counts)/np.mean(counts)) #Fano-factor
    print(f"Step {i}")

# %%
#Plot for mean and variance, with fit
# coeffs = np.polyfit(mean, variance, 1)  # [slope, intercept]
# fit = np.poly1d(coeffs)

x_fit = np.linspace(0, 240, 100) 

plt.figure(dpi=150)
# plt.plot(x_fit,fit(x_fit),label=f'Linear fit: Fano factor {fit[1]:.3f}')#fit did not go through origin
plt.plot(x_fit, np.mean(F)*x_fit, label=f'F={np.mean(F):.3f}') #line y=Fx
plt.axis((0, 245, 0, 6))
plt.errorbar(mean, variance, xerr=mean_err, yerr=var_err, fmt='o', 
             capsize=3, label=f'Measured data (varying exposure time)')
plt.title("Mean-variance Relation")
plt.xlabel("Mean (ADU)")
plt.ylabel("Variance (ADU²)")
plt.legend()
plt.show()


