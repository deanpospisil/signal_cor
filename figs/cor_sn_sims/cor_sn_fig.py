#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:32:54 2020

@author: dean
"""

import matplotlib.pyplot as plt

import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.patches import Ellipse

def fz(r):
    return 0.5*(np.log((1+r)/(1-r)))

def inv_fz(z):
    return (np.exp(2*z) - 1)/(np.exp(2*z) + 1)

def noise_cor_fast(x, y):
    #n,m,sim
    X_ms = x-x.mean(0)
    X_ms_n = X_ms/(X_ms**2.).sum(0)**0.5
    
    Y_ms = y-y.mean(0)
    Y_ms_n = Y_ms/(Y_ms**2.).sum(0)**0.5
    
    hat_r_n = (Y_ms_n*X_ms_n).sum(0).mean(0)

    return hat_r_n

def sig_cor_fast(x, y):
    #n,m,sim
    X_m = x.mean(0)
    Y_m = y.mean(0)
    
    Y_m_ms = Y_m-Y_m.mean(0)
    Y_m_ms_n = Y_m_ms/(Y_m_ms**2).sum(0)**0.5
    
    X_m_ms = X_m-X_m.mean(0)
    X_m_ms_n = X_m_ms/(X_m_ms**2).sum(0)**0.5
    
    hat_r_s = (X_m_ms_n*Y_m_ms_n).sum(0)
    
    return hat_r_s


def n2n_pearson_r_sin(r, m):

    angle = np.arccos(r)[np.newaxis, :]  
    s = np.linspace(0, 2*np.pi-2*np.pi/m, int(m))[:, np.newaxis, np.newaxis]
   
    xu = np.cos(s + angle*0)
    yu = np.cos(angle + s)

    yu = (yu/((yu**2.).sum(0)**0.5))
    xu = (xu/((xu**2.).sum(0)**0.5))

   
    return np.array([xu, yu])

def gen_a_sig_noise_cor_sim_fixed_stim(r_s, r_n, n_sims, 
                                     snr, n, m,
                                     n_x, n_y,  a_x, a_y):
    
    #fixed
    n_y = np.sign(r_n)*n_y
    
    #derived
    d_x = (m*snr*np.abs(n_x*n_y))**0.5
    d_y = d_x.copy()
    
    
    X_s,Y_s = n2n_pearson_r_sin(r_s, m)
    X_s = a_x + d_x*X_s
    Y_s = a_y + d_y*Y_s
    
    C = np.random.normal(0, 1, size=(n, m, p_cell_pairs, n_sims))
    N_x = np.random.normal(0, 1, size=(n, m, p_cell_pairs, n_sims))
    N_y = np.random.normal(0, 1, size=(n, m, p_cell_pairs, n_sims))
    
    X_n =  n_x*((abs(r_n)**0.5)*C + ((1-abs(r_n))**0.5)*N_x)
    X = X_s + X_n
    
    Y_n =    n_y*((abs(r_n)**0.5)*C + ((1-abs(r_n))**0.5)*N_y)
    Y = Y_s + Y_n
    
    hat_r_n = noise_cor_fast(X, Y)
   
    hat_r_s = sig_cor_fast(X, Y)
    
    
    return Y_s, X_s, Y, X, hat_r_n, hat_r_s



#%% fixed stim
ylim=(-0.1, 1)
fs = 12

yticks =  np.linspace(0, 1, 5)

fns = [  'rho_sn_vs_m_snr_hi_fxd_stim.nc','rho_sn_vs_m_snr_low_fxd_stim.nc',
       'rho_sn_vs_n_fxd_stim.nc', 'rho_sn_vs_SNR_fxd_stim.nc', 'rho_sn_vs_SNR_split_fxd_stim.nc', ]

das = [xr.open_dataarray(fn) for fn in fns]

fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(8,3))
panel_labels = ['A.', 'B.', 'C.', 'D.', 'E.']
xlabels = ['m', 'm', 'n', 'SNR', 'SNR']
titles = ['SNR=1000', 'SNR=0.1', 'm=100', '', 'Split-half $\hat{r}_{signal}$', ] 
(s_x,s_y) = (-0.1, 1.2)
for i, da in enumerate(das):
    dims = da.dims
    dim = [a for a in dims if len(da.coords[a])>1 and not 'rho' in a][0]
    coords = da.coords[dim].values
    rho_rs_rns = da.coords['rho']
    n_sims = len(da.coords['sim'])
    
    for a_rho in rho_rs_rns:
        axs[i].errorbar(da.coords[dim], 
                     da.sel(rho=a_rho).mean('sim').squeeze(), 
                     yerr=(da.sel(rho=a_rho).std('sim')/n_sims**0.5).squeeze())
    
    axs[i].set_yticks(yticks)
    axs[i].set_ylim(ylim)
    if not i==2:
        axs[i].semilogx()
        
    axs[i].set_xticks(coords)
    print(coords)
    axs[i].set_xlabel(xlabels[i])
    axs[i].grid()
    if not i==0:
        axs[i].set_yticklabels([])
    else:
        axs[i].set_ylabel(r'Corr($\hat{r}_{s}$, $\hat{r}_{n}$)')
        axs[i].legend(da.coords['rho'].values, fontsize=6, 
                      title=r'$\rho_{SN}$', loc='upper left')
    axs[i].set_title(titles[i])
    axs[i].annotate(panel_labels[i], (s_x, s_y), 
             annotation_clip=False, fontsize=fs,
             weight='bold',
             xycoords='axes fraction')
    
    if i<2:
        axs[i].set_xticklabels([10, 100, 1000])
    if i>2:
        axs[i].set_xticklabels([0.1, 1, 10, 100])
    
    if i==2:
        axs[i].semilogx(subsx=[0])
        axs[i].set_xticks([4,8, 16, 32, 64])
        axs[i].set_xticklabels([4,8, 16, 32, 64])
        ''
fig.tight_layout()        
plt.savefig('rho_sn_vs_all.pdf')
    
#%% deflation of corr of sig cor and noise cor 
from common_cor_sim import hat_r_s_naive_fixed_stim, hat_r_s_naive_rand_stim
from scipy.stats import spearmanr
(s_x,s_y) = (-1.6, 1.5)
ticks = np.linspace(1,-1,5)
fs = 12
tick_labels = ['1','','0', '', '-1']

rho_rs_rn = 0.8
nm = 'example_sim_fxd_stim_rn='+ str(rho_rs_rn)+'.nc'
d = xr.open_dataset(nm)
snr = d.attrs['snr']
m = d.attrs['m']
n = d.attrs['m']
r_s = d.attrs['u_s']
n_x = n_y = d.attrs['n_x']
d_x = d_y = (snr*m)**0.5
hat_r_s_pred = []
r_ns = np.linspace(-.75, .75)
for r_n in r_ns:
    hat_r_s_pred.append(hat_r_s_naive_fixed_stim(r_n, r_s, d_x, d_y, n_x, n_y, n, m))
    
sc = .7
plt.figure(figsize=((9, 4)))
r_s = d['r_s']
r_n = d['r_n']
X_s = d['X_s']
Y_s = d['Y_s']
X = d['X']
Y = d['Y']
ind = 2
hat_r_s = d['hat_r_s']
hat_r_n = d['hat_r_n']


a_rs = r_s[:,ind]
a_rn = r_n.squeeze()[:,ind]
s=2

plt.subplot(151)
plt.scatter(a_rn, a_rs, s=s);
plt.xlabel(r'$r_n$', fontsize=fs)
plt.ylabel(r'$r_s$', fontsize=fs)
plt.axis('square')
plt.xticks(ticks)
plt.yticks(ticks)
plt.xlim(-1,1);plt.ylim(-1,1)
plt.grid()
plt.gca().set_xticklabels(tick_labels)
plt.gca().set_yticklabels(tick_labels)
plt.annotate('A.', (s_x, s_y), 
             annotation_clip=False, fontsize=fs, weight='bold')


plt.title(r'Corr($r_s, \rho_n$)=' + str(rho_rs_rn) + '\n' + 
          'SNR=' +str(snr))
plt.subplot(152)
plt.plot(r_ns, hat_r_s_pred)
plt.scatter(a_rn, hat_r_s[:,ind], s=s)

plt.axis('square')

plt.xticks(ticks)
plt.yticks(ticks)
plt.xlim(-1,1);plt.ylim(-1,1)

#plt.xlim(0,1);plt.ylim(0,1)

plt.grid()
plt.title(r'Corr($\hat{r}_s, {r}_n$)=' + str(np.round(np.corrcoef(a_rn, hat_r_s[:,ind])[0,1], 2)))

plt.ylabel(r'$\hat{r}_s$', fontsize=fs)
plt.xlabel(r'$r_n$', fontsize=fs)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])

plt.annotate('B.', (s_x, s_y), 
             annotation_clip=False, fontsize=fs, weight='bold')

plt.subplot(153)

plt.scatter(a_rs, hat_r_s[:,ind], s=s)

plt.axis('square')
plt.xticks(ticks)
plt.yticks(ticks)
plt.xlim(-1,1);plt.ylim(-1,1)
plt.grid()
plt.ylabel(r'$\hat{r}_s$', fontsize=fs)
plt.xlabel(r'$r_s$', fontsize=fs)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plt.annotate('C.', (s_x, s_y), 
             annotation_clip=False, fontsize=fs, weight='bold')
plt.subplot(154)

plt.scatter(a_rn, hat_r_n[:,ind], s=s)

plt.axis('square')
plt.xticks(ticks)
plt.yticks(ticks)
plt.xlim(-1,1);plt.ylim(-1,1)
plt.grid()
plt.ylabel(r'$\hat{r}_n$', fontsize=fs)
plt.xlabel(r'$r_n$', fontsize=fs)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plt.annotate('D.', (s_x, s_y), 
             annotation_clip=False, fontsize=fs, weight='bold')
plt.subplot(155)

plt.scatter(hat_r_n[:,ind], hat_r_s[:,ind], s=s)
plt.axis('square')
plt.xticks(ticks)
plt.yticks(ticks)
plt.xlim(-1,1);plt.ylim(-1,1)
plt.xlabel(r'$\hat{r}_n$', fontsize=fs)
plt.ylabel(r'$\hat{r}_s$', fontsize=fs)
plt.grid()
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plt.title(r'Corr($\hat{r}_s, \hat{r}_n$)=' + str(np.round(np.corrcoef(hat_r_s[:,ind], hat_r_n[:,ind])[0,1], 2)))
plt.annotate('E.', (s_x, s_y), 
             annotation_clip=False, fontsize=fs, weight='bold')
plt.tight_layout()
plt.savefig('example_defl_rs_rn.pdf')

#%% 


rho_rs_rn = 0
nm = 'example_sim_fxd_stim_rn='+ str(rho_rs_rn)+'.nc'
d = xr.open_dataset(nm)
snr = d.attrs['snr']
m = d.attrs['m']
n = d.attrs['m']
r_s = d.attrs['u_s']
n_x = n_y = d.attrs['n_x']
d_x = d_y = (snr*m)**0.5
hat_r_s_pred = []
r_ns = np.linspace(-.75, .75)
for r_n in r_ns:
    hat_r_s_pred.append(hat_r_s_naive_fixed_stim(r_n, r_s, d_x, d_y, n_x, n_y, n, m))
    
sc = .7
plt.figure(figsize=((9, 4)))
r_s = d['r_s']
r_n = d['r_n']
X_s = d['X_s']
Y_s = d['Y_s']
X = d['X']
Y = d['Y']
ind = 0
hat_r_s = d['hat_r_s']
hat_r_n = d['hat_r_n']

a_rs = r_s[:,ind]
a_rn = r_n.squeeze()[:,ind]
s=2

plt.subplot(151)
plt.scatter(a_rn, a_rs, s=s);
plt.xlabel(r'$r_n$', fontsize=fs)
plt.ylabel(r'$r_s$', fontsize=fs)
plt.axis('square')
plt.xticks(ticks)
plt.yticks(ticks)
plt.xlim(-1,1);plt.ylim(-1,1)
plt.grid()
plt.gca().set_xticklabels(tick_labels)
plt.gca().set_yticklabels(tick_labels)
plt.annotate('A.', (s_x, s_y), 
             annotation_clip=False, fontsize=fs, weight='bold')


plt.title(r'Corr($r_s, \rho_n$)=' + str(rho_rs_rn) + '\n' + 
          'SNR=' +str(snr))
plt.subplot(152)
plt.plot(r_ns, hat_r_s_pred)
plt.scatter(a_rn, hat_r_s[:,ind], s=s)

plt.axis('square')

plt.xticks(ticks)
plt.yticks(ticks)
plt.xlim(-1,1);plt.ylim(-1,1)

#plt.xlim(0,1);plt.ylim(0,1)

plt.grid()
plt.title(r'Corr($\hat{r}_s, {r}_n$)=' + str(np.round(np.corrcoef(a_rn, hat_r_s[:,ind])[0,1], 2)))

plt.ylabel(r'$\hat{r}_s$', fontsize=fs)
plt.xlabel(r'$r_n$', fontsize=fs)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])

plt.annotate('B.', (s_x, s_y), 
             annotation_clip=False, fontsize=fs, weight='bold')

plt.subplot(153)

plt.scatter(a_rs, hat_r_s[:,ind], s=s)

plt.axis('square')
plt.xticks(ticks)
plt.yticks(ticks)
plt.xlim(-1,1);plt.ylim(-1,1)
plt.grid()
plt.ylabel(r'$\hat{r}_s$', fontsize=fs)
plt.xlabel(r'$r_s$', fontsize=fs)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plt.annotate('C.', (s_x, s_y), 
             annotation_clip=False, fontsize=fs, weight='bold')
plt.subplot(154)

plt.scatter(a_rn, hat_r_n[:,ind], s=s)

plt.axis('square')
plt.xticks(ticks)
plt.yticks(ticks)
plt.xlim(-1,1);plt.ylim(-1,1)
plt.grid()
plt.ylabel(r'$\hat{r}_n$', fontsize=fs)
plt.xlabel(r'$r_n$', fontsize=fs)
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plt.annotate('D.', (s_x, s_y), 
             annotation_clip=False, fontsize=fs, weight='bold')
plt.subplot(155)

plt.scatter(hat_r_n[:,ind], hat_r_s[:,ind], s=s)
plt.axis('square')
plt.xticks(ticks)
plt.yticks(ticks)
plt.xlim(-1,1);plt.ylim(-1,1)
plt.xlabel(r'$\hat{r}_n$', fontsize=fs)
plt.ylabel(r'$\hat{r}_s$', fontsize=fs)
plt.grid()
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plt.title(r'Corr($\hat{r}_s, \hat{r}_n$)=' + str(np.round(np.corrcoef(hat_r_s[:,ind], hat_r_n[:,ind])[0,1], 2)))
plt.annotate('E.', (s_x, s_y), 
             annotation_clip=False, fontsize=fs, weight='bold')
plt.tight_layout()
plt.savefig('example_infl_rs_rn.pdf')

#%%



rho_rs_rn = 0.8
n_sims = 100
p_cell_pairs = 50

snr = 1
n = 3
m = 8
n_x = 1
n_y = 1
a_x = 6
a_y = 5.5
r_s = np.array([0.3,])
r_n = np.array([0.9,])

Y_s, X_s, Y, X, hat_r_n, hat_r_s = gen_a_sig_noise_cor_sim_fixed_stim(
                                     r_s, r_n, n_sims, 
                                     snr, n, m,
                                     n_x, n_y,  a_x, a_y, )

ind = 0 
plt.figure(figsize=(8,3))
r_noise = np.round(float(r_n),2)

sig2 = 1
r2=0.5
angle = np.arccos(r2)
#s = np.linspace(0, 2*np.pi- 2*np.pi/m, m)
mu2x=mu2y=20


leg_fs = 6

sig2 = 1
plt.subplot(121)

#plt.title('Neuron to neuron simulation')

xu = X_s.squeeze()
yu = Y_s.squeeze()
angle = np.arccos(r2**0.5)
m = len(xu)

s = np.linspace(0, 2*np.pi-2*np.pi/m, m)


plt.plot(s, xu, 'b-', alpha=0.5)
plt.plot(s, yu, 'r-', alpha=0.5)
n = X.shape[0]


plt.plot(s, xu, 'bo', alpha=0.5, ms=0)
plt.plot(s, yu, 'ro', alpha=0.5, ms=0)

plt.errorbar(s, yu, yerr=(sig2/n)**0.5, c='r', linestyle='', marker='.', alpha=0.5)
plt.errorbar(s, xu, yerr=(sig2/n)**0.5, c='b', linestyle='', marker='.', alpha=0.5)


xm = X[:, :, ind, 0].mean(0)
ym = Y[:,:, ind, 0].mean(0)

plt.plot(s, xm, 'b:o', mfc='none',)
plt.plot(s, ym, 'r:o', mfc='none',)



plt.ylabel(r'$ \sqrt{{ \mathrm{spike \ count}}}$')
plt.xlabel('Stimuli phase (deg)')
plt.xticks([0, np.pi, np.pi*2])
plt.gca().set_xticklabels([r'$0$',r'$\pi$',r'$2\pi$',])
plt.gca().set_xticklabels([r'$0$',r'$180$',r'$360$',])

ms =4
legend_elements = [
                   Line2D([0], [0], marker='o', color='r', 
                          label='Neuron X expected response',
                          markerfacecolor='r', markersize=ms, lw=2),
                Line2D([0], [0], marker='o',lw=2, color='r', 
                       label='Neuron X trial avg.',
                                 mfc='none', markersize=ms,linestyle=':'),
                # Line2D([0], [0], marker='.',lw=2, color='r', 
                #        label='Neuron X single trial ',
                #                  mfc='r', markersize=ms,linestyle=''),
                   Line2D([0], [0], marker='o', color='b', lw=2,
                          label='Neuron Y expected response',
                          mfc='b', markersize=ms),

                    Line2D([0], [0], marker='o',lw=2, color='b', 
                           label='Neuron X trial avg.',
                          mfc='none', markersize=ms, linestyle=':'),
                   # Line2D([0], [0], marker='.',lw=2, color='b', 
                   #     label='Neuron X single trial ',
                   #               mfc='b', markersize=ms,linestyle='')
                ]
plt.xlim(-2*np.pi/m*1.5, np.pi*2*1.1)
plt.ylim(0,10)
plt.gca().set_yticks(range(0,10,2))
#plt.legend(handles=legend_elements, loc='lower left', fontsize=leg_fs)
plt.text(-1, 8.5, r'True $r_{\mathrm{s}}=$'+
         str(np.round(float(r_s),2))+ 
         r', Measured $\hat{r}_{\mathrm{s}}=$'+
         str(np.round(float(hat_r_s[ind,0]),2)))

plt.text(-1, 1, r'$n=$'+str(n) + 
         r'$, m=$' + str(m))
         

#plt.legend(handles=legend_elements, fontsize=leg_fs, loc='lower right' ,
#           labelspacing=0.4)
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
                    angle=45,
                    height=0.5, ec='k', fc='none', lw=1)

    ax.add_artist(e)
plt.axis('square')
plt.xlim(0,10);
plt.ylim(0,10);
plt.gca().set_xticks(range(0,10,2))
plt.gca().set_yticks(range(0,10,2))
plt.xlabel('Neuron X response \n $ \sqrt{{ \mathrm{spike \ count}}}$')
plt.ylabel('Neuron Y response \n $ \sqrt{{ \mathrm{spike \ count}}}$')
#plt.title('Joint distribution individual trials')
plt.text(.1, 9, r'True $r_{\mathrm{n}}=$' + str(r_noise))
#plt.text(.1, 7.8, r'Measured $\hat{r}_{\mathrm{n}}=$' 
#         + str(np.round(hat_r_n[ind,0], 2)))
plt.tight_layout()

plt.savefig('example_pair_schematic_infl.pdf')


#%%
snr = 1
n = 3
m = 8
n_x = 1
n_y = 1
a_x = 6
a_y = 5.5
r_s = np.array([0.5,])
r_n = np.array([0.001,])

Y_s, X_s, Y, X, hat_r_n, hat_r_s = gen_a_sig_noise_cor_sim_fixed_stim(
                                     r_s, r_n, n_sims, 
                                     snr, n, m,
                                     n_x, n_y,  a_x, a_y, )

ind = 0 
plt.figure(figsize=(8,3))
r_noise = np.round(float(r_n),2)

sig2 = 1
angle = np.arccos(r2)
#s = np.linspace(0, 2*np.pi- 2*np.pi/m, m)
mu2x=mu2y=20


leg_fs = 6

sig2 = 1
plt.subplot(121)

#plt.title('Neuron to neuron simulation')

xu = X_s.squeeze()
yu = Y_s.squeeze()
angle = np.arccos(r2**0.5)
m = len(xu)

s = np.linspace(0, 2*np.pi-2*np.pi/m, m)


plt.plot(s, xu, 'b-', alpha=0.5)
plt.plot(s, yu, 'r-', alpha=0.5)
n = X.shape[0]


plt.plot(s, xu, 'bo', alpha=0.5, ms=0)
plt.plot(s, yu, 'ro', alpha=0.5, ms=0)

plt.errorbar(s, yu, yerr=(sig2/n)**0.5, c='r', linestyle='', marker='.', alpha=0.5)
plt.errorbar(s, xu, yerr=(sig2/n)**0.5, c='b', linestyle='', marker='.', alpha=0.5)


xm = X[:, :, ind, 0].mean(0)
ym = Y[:,:, ind, 0].mean(0)

plt.plot(s, xm, 'b:o', mfc='none',)
plt.plot(s, ym, 'r:o', mfc='none',)



plt.ylabel(r'$ \sqrt{{ \mathrm{spike \ count}}}$')
plt.xlabel('Stimuli phase (deg)')
plt.xticks([0, np.pi, np.pi*2])
plt.gca().set_xticklabels([r'$0$',r'$\pi$',r'$2\pi$',])
plt.gca().set_xticklabels([r'$0$',r'$180$',r'$360$',])

ms =4
legend_elements = [
                   Line2D([0], [0], marker='o', color='r', 
                          label='Neuron X expected response',
                          markerfacecolor='r', markersize=ms, lw=2),
                Line2D([0], [0], marker='o',lw=2, color='r', 
                       label='Neuron X trial avg.',
                                 mfc='none', markersize=ms,linestyle=':'),
                # Line2D([0], [0], marker='.',lw=2, color='r', 
                #        label='Neuron X single trial ',
                #                  mfc='r', markersize=ms,linestyle=''),
                   Line2D([0], [0], marker='o', color='b', lw=2,
                          label='Neuron Y expected response',
                          mfc='b', markersize=ms),

                    Line2D([0], [0], marker='o',lw=2, color='b', 
                           label='Neuron X trial avg.',
                          mfc='none', markersize=ms, linestyle=':'),
                   # Line2D([0], [0], marker='.',lw=2, color='b', 
                   #     label='Neuron X single trial ',
                   #               mfc='b', markersize=ms,linestyle='')
                ]
plt.xlim(-2*np.pi/m*1.5, np.pi*2*1.1)
plt.ylim(0,10)
plt.gca().set_yticks(range(0,10,2))
#plt.legend(handles=legend_elements, loc='lower left', fontsize=leg_fs)
plt.text(-1, 8.5, r'True $r_{\mathrm{s}}=$'+
         str(np.round(float(r_s),2))+ 
         r', Measured $\hat{r}_{\mathrm{s}}=$'+
         str(np.round(float(hat_r_s[ind,0]),2)))

plt.text(-1, 1, r'$n=$'+str(n) + 
         r'$, m=$' + str(m))
         

plt.legend(handles=legend_elements, fontsize=leg_fs, loc='lower right' ,
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
                    width=1,
                    angle=0,
                    height=1, ec='k', fc='none', lw=1)

    ax.add_artist(e)
plt.axis('square')
plt.xlim(0,10);
plt.ylim(0,10);
plt.gca().set_xticks(range(0,10,2))
plt.gca().set_yticks(range(0,10,2))
plt.xlabel('Neuron X response \n $( \sqrt{{ \mathrm{spike \ count}}})$')
plt.ylabel('Neuron Y response  \n $( \sqrt{{ \mathrm{spike \ count}}})$')
#plt.title('Joint distribution individual trials')
plt.text(.1, 9, r'True $r_{\mathrm{n}}=$' + str(r_noise))
#plt.text(.1, 7.8, r'Measured $\hat{r}_{\mathrm{n}}=$' 
#         + str(np.round(hat_r_n[ind,0], 2)))
plt.tight_layout()

plt.savefig('example_pair_schematic_defl.pdf')

#%% analytic relationship between r_n and hat_r_s
from common_cor_sim import hat_r_s_naive_fixed_stim, hat_r_s_naive_rand_stim

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
m = 9
snrs = np.array([0.1, 0.5, 1])
n_s = np.array([3, 6, 9])
r_s_s = np.array([0, 0.25, 0.75])
r_n_s = np.array([0, 0.25, 0.75])
r_n_s = np.linspace(-1,1, 30)
params_list = [snrs, n_s, r_s_s, r_n_s]
param_shape = tuple(len(param) for param in params_list)

da = xr.DataArray(np.zeros(param_shape),
             coords=params_list,
             dims=['snr', 'n', 'r_s', 'r_n'])
das = []
colors = ['purple', 'red', 'orange']
for rand in list(range(2))[:1]:

    da[...] = 0 
    from itertools import product
    p = product(*params_list)
    
    for ap in p:
        snr, n, r_s, r_n = ap
        #print(ap)
        d_x = d_y = snr**0.5
        n_x = n_y = 1
        if rand==1:
            d_x = d_y = (snr*m)**0.5
            hat_r_s = hat_r_s_naive_fixed_stim(r_n, r_s, d_x, d_y, n_x, n_y, n, m)
        else:
            hat_r_s = hat_r_s_naive_rand_stim(r_n, r_s, d_x, d_y, n_x, n_y, n)
        da.loc[snr, n, r_s, r_n] = hat_r_s
    das.append(da)
    plt.figure(figsize=(5.6,5.6))
    ticks = np.linspace(-1,1,9)
    #tick_labels = ['' , '-0.5', '' , '-1'][::-1]+['0', '', '0.5', '', '1']
    for i in range(3):
        for j in range(3):

            dat = da[:, i, j]
            k = np.ravel_multi_index((i,j), dims=(3,3))
            plt.subplot(3,3,k+1)

            
            for q in range(3):
                plt.plot(dat.coords['r_n'].values, dat[q], color=colors[q])
            plt.axis('square')
            plt.ylim(-1,1)
            plt.xlim(-1,1)
            plt.xticks(ticks)
            plt.yticks(ticks)
            #plt.grid()
            
            if i==0:
                plt.title('\n \n' + '$r_s$=' + str(dat.coords['r_s'].values));
            if j==2:
                plt.ylabel('$n=$' + str(dat.coords['n'].values), rotation=0, 
                           labelpad=20)
                ax = plt.gca()
                ax.yaxis.set_label_position("right")
            if k==0:
                plt.legend(snrs, title='SNR', loc='lower right', fontsize=7)
            if not (j==0 and i==2):
                plt.gca().set_xticklabels([]);plt.gca().set_yticklabels([]);
                
            else:
                #plt.xlabel( r'$r_n$'+ '\n' + '$r_s$=' + str(dat.coords['r_s'].values));
                plt.xlabel( r'$r_n$');
                
                #plt.ylabel('$n=$' + str(dat.coords['n'].values) + '\n' + r'$\hat{r}_s$')
                plt.ylabel('\n' + r'$\hat{r}_s$')
                #plt.gca().set_xticklabels(tick_labels);
                #plt.gca().set_yticklabels(tick_labels);
                
            # plt.title('n='+str(dat.coords['n'].values) + ', ' +
            #           r'$r_s$='+str(dat.coords['r_s'].values))
                
            plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))
            plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))
            plt.gca().xaxis.set_minor_locator(MultipleLocator(0.25))
            plt.gca().yaxis.set_minor_locator(MultipleLocator(0.25))

            plt.plot([-1,1], [0,0], c='k', lw=1, alpha=0.25)
            plt.plot([0,0], [-1,1], c='k', lw=1, alpha=0.25)
            plt.gca().tick_params(which='minor', length=2)
            plt.grid(which='both')

    plt.tight_layout(h_pad=-1, w_pad=0) 
    plt.savefig('hat_r_s_r_n_rel_anlytc_rand='+ str(rand) +'_.pdf')        
#%%
da = xr.open_dataarray('sim_r2s_est_comp.nc')
da_rn = da.rename({'r_s':r'$r_s$', 'snr':'SNR', 'r_n':r'$r_n$', 'est':'Estimate'})
da_rn.coords[r'$r_s$'] = da_rn.coords[r'$r_s$']**2
da_rn.coords['Estimate'] = [r'$\hat{r}^2$', r'$\hat{r}^2_{split}$', 
                            r'$\hat{r}^2_{ER}$', r'$\hat{r}^2_{ER_{split}}$']

g = da_rn.mean('sim').sel(n=8, m=500, SNR=[0.1,0.5,1]).plot(x=r'$r_s$', 
                                                              hue='Estimate',
                                                              col=r'$r_n$', 
                                                              row='SNR', 
                                                              sharex=False,
                                                              sharey=False,
                                                              add_legend=False)

da_rn_m = da_rn.mean('sim').sel(n=8, m=500, SNR=[0.1,0.5,1])
da_rn_sd = da_rn.std('sim').sel(n=8, m=500, SNR=[0.1,0.5,1])

ticks = np.linspace(0, 1, 5)
colors = ['blue', 'orange', 'g', 'r']
for i, ax_r in enumerate(g.axes):
    for j, ax in enumerate(ax_r):
        ax.grid()
        ax.axis('square')
        ax.set_ylim(-0.1,1.1)
        ax.set_xlim(-0.1,1.1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        if j==0 and i==2:
            ax.set_xlabel('True $r^2_s$');
            ax.set_ylabel('Estimated $\hat{r}_s$')
            ax.legend([r'$\hat{r}^2$', r'$\hat{r}^2_{split}$', 
                            r'$\hat{r}^2_{ER}$', r'$\hat{r}^2_{ER_{split}}$'])
        else:
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        for k in range(4):
            ax.errorbar(x=da_rn_m.coords[r'$r_s$'], 
                        y=da_rn_m[i,:,j, k],
                        yerr=da_rn_sd[i,:,j, k],
                        c=ax.lines[k].get_c())
            

plt.tight_layout()

plt.savefig('sim_r2s_est_comp.pdf')



#%%
#%% fixed stim
ylim=(-0.1, 1)
fs = 12

yticks =  np.linspace(0, 1, 5)

fns = [  'rho_sn_vs_m_snr_hi_fxd_stim.nc','rho_sn_vs_m_snr_low_fxd_stim.nc',
       'rho_sn_vs_n_fxd_stim.nc', 'rho_sn_vs_SNR_fxd_stim.nc',  ]

das = [xr.open_dataarray(fn) for fn in fns]

fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(8,3))
colors = plt.cm.autumn_r(np.linspace(0,1,4))
panel_labels = ['A.', 'B.', 'C.', 'D.', 'E.']
xlabels = ['m', 'm', 'n', 'SNR', 'SNR']
titles = ['SNR=1000', 'SNR=0.1', 'm=100', '', 'Split-half $\hat{r}_{signal}$', ] 
(s_x,s_y) = (-0.1, 1.2)
for i, da in enumerate(das):
    for split in [0, 1]:
        a_da = da.sel(split=split)
        dims = a_da.dims
        dim = [a for a in dims if len(a_da.coords[a])>1 and not 'rho' in a][0]
        coords = a_da.coords[dim].values
        rho_rs_rns = a_da.coords['rho']
        n_sims = len(a_da.coords['sim'])
        
        if split==0:
            alpha=1
        else:
            alpha=0.5
            
        for k, a_rho in enumerate(rho_rs_rns):
            axs[i].errorbar(a_da.coords[dim], 
                         a_da.sel(rho=a_rho).mean('sim').squeeze(), 
                         yerr=(a_da.sel(rho=a_rho).std('sim')/n_sims**0.5).squeeze(),
                         alpha=alpha,color=colors[k])
        
    axs[i].set_yticks(yticks)
    axs[i].set_ylim(ylim)
    if not i==2:
        axs[i].semilogx()
        
    axs[i].set_xticks(coords)
    print(coords)
    axs[i].set_xlabel(xlabels[i])
    axs[i].grid()
    if not i==0:
        axs[i].set_yticklabels([])
    else:
        axs[i].set_ylabel(r'Corr($\hat{r}_{s}$, $\hat{r}_{n}$)')
        axs[i].legend(a_da.coords['rho'].values, fontsize=6, 
                      title=r'$\rho_{SN}$', loc='upper left')
    axs[i].set_title(titles[i])
    axs[i].annotate(panel_labels[i], (s_x, s_y), 
             annotation_clip=False, fontsize=fs,
             weight='bold',
             xycoords='axes fraction')

    if i<2:
        axs[i].set_xticklabels([10, 100, 1000])
    if i>2:
        axs[i].set_xticklabels([0.1, 1, 10, 100])
    
    if i==2:
        axs[i].semilogx(subsx=[0])
        axs[i].set_xticks([4,8, 16, 32, 64])
        axs[i].set_xticklabels([4,8, 16, 32, 64])
        ''
fig.tight_layout()        
plt.savefig('rho_sn_vs_all.pdf')