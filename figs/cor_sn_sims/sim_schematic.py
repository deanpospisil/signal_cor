#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:30:39 2020

@author: dean
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import os
import xarray as xr
from scipy.stats import spearmanr
from matplotlib.lines import Line2D

my = 7
mx = np.pi*2
ms = 6
r2 = 0.3
m=8
n=5

sig2 = 0.25
angle = np.arccos(r2**0.5)
s = np.linspace(0, 2*np.pi- 2*np.pi/m, m)
mu2x=mu2y=20


leg_fs = 6
plt.figure(figsize=(8,3))

sig2 = 1
plt.subplot(121)

angle = np.arccos(r2**0.5)
s = np.linspace(0, 2*np.pi, m)
s = np.linspace(0, 2*np.pi, m*100)
mu2x=20*90
mu2y =5*90
xu = np.sin(s)
yu = np.sin(angle + s)
xu = (mu2x**0.5)*(xu/(xu**2.).sum(-1)**0.5)
yu = (mu2y**0.5)*(yu/(yu**2.).sum(-1)**0.5)
xu = xu-2*xu.min()
yu = yu +  xu.mean()


plt.title('Neuron to neuron simulation')


plt.plot(s, xu, 'b-', alpha=0.5)
plt.plot(s, yu, 'r-', alpha=0.5)

xu = xu[::100]
yu = yu[::100]
s = s[::100]
plt.plot(s, xu, 'bo', alpha=0.5, ms=0)
plt.plot(s, yu, 'ro', alpha=0.5, ms=0)

plt.errorbar(s,yu,yerr=0.25**0.5, c='r', linestyle='', marker='.', alpha=0.5)
plt.errorbar(s,xu,yerr=0.25**0.5, c='b', linestyle='', marker='.', alpha=0.5)


x = np.random.normal(loc=xu, scale=sig2**0.5,
                     size=(int(n),) + xu.shape )
y = np.random.normal(loc=yu, scale=sig2**0.5,
                     size= (int(n),) + yu.shape )

xm = x.mean(0)
ym = y.mean(0)

plt.plot(s, xm, 'b:o', mfc='none',)
plt.plot(s, ym, 'r:o', mfc='none',)

s = np.broadcast_to(s, np.shape(y))

#plt.plot(s.ravel(), x.ravel(), '.', c='b', alpha=1)
#plt.plot(s.ravel(), y.ravel(), '.', c='r', alpha=1)


plt.ylabel(r'$ \sqrt{{ \mathrm{spike \ count}}}$')
plt.xlabel('Stimuli phase (Deg.)')
plt.xticks([0, np.pi, np.pi*2])
plt.gca().set_xticklabels([r'$0$',r'$\pi$',r'$2\pi$',])
plt.gca().set_xticklabels([r'$0$',r'$180$',r'$360$',])


legend_elements = [
                   Line2D([0], [0], marker='o', color='r', 
                          label='Neuron X expected response',
                          markerfacecolor='r', markersize=ms, lw=2),
                Line2D([0], [0], marker='o',lw=2, color='r', 
                       label='Neuron X trial avg.',
                                 mfc='none', markersize=ms,linestyle=':'),
                Line2D([0], [0], marker='.',lw=2, color='r', 
                       label='Neuron X single trial ',
                                 mfc='r', markersize=ms,linestyle=''),
                   Line2D([0], [0], marker='o', color='b', lw=2,
                          label='Neuron Y expected response',
                          mfc='b', markersize=ms),

                    Line2D([0], [0], marker='o',lw=2, color='b', 
                           label='Neuron X trial avg.',
                          mfc='none', markersize=ms, linestyle=':'),
                   Line2D([0], [0], marker='.',lw=2, color='b', 
                       label='Neuron X single trial ',
                                 mfc='b', markersize=ms,linestyle=''),]
plt.xlim(-2*np.pi/m*1.5, mx*1.1)
plt.ylim(0,my*1)
plt.gca().set_yticks(range(0,7,2))
#plt.legend(handles=legend_elements, loc='lower left', fontsize=leg_fs)
plt.text(np.pi/2*2.2, 4.7, r'True $r^2_{\mathrm{signal}}=$'+str(r2)+  '\n' r'$n=$'+str(n)+ '\n' + 
         r'$m=$' + str(m))
plt.legend(handles=legend_elements, fontsize=leg_fs, loc='lower left' ,
           labelspacing=0.4)
plt.subplots_adjust(hspace=.2)
plt.subplot(122)

from matplotlib import cm
x = np.linspace(0,1,9)
rgb = cm.get_cmap('Set1')(x)[ :, :3]

plt.plot(xu, yu, '.', mfc='k', c='k')
from matplotlib.patches import Ellipse
ax = plt.gca()

for i in range(m):
    e = Ellipse(xy=(xu[i], yu[i]),
                    width=2,
                    angle=30,
                    height=1, ec='k', fc='none', lw=1)

    ax.add_artist(e)
plt.axis('square')
plt.xlim(0,7.5);
plt.ylim(0,7.5);
plt.gca().set_xticks(range(0,7,2))
plt.gca().set_yticks(range(0,7,2))
plt.xlabel('Neuron X single trial \n $ \sqrt{{ \mathrm{spike \ count}}}$')
plt.ylabel('Neuron Y single trial \n $ \sqrt{{ \mathrm{spike \ count}}}$')
plt.title('Joint distribution individual trials')
plt.text(.5,6, r'True $r^2_{\mathrm{noise}}=0.5$')
plt.tight_layout()
plt.savefig('n2n_simulation_schematic.pdf')

#%%
