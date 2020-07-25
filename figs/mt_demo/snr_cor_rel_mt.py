#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 11:52:49 2020

@author: dean
"""
import numpy as np
import xarray as xr
import pandas as pd
import os
os.chdir('../../')
import r2c_common as rc
from scipy import stats
import matplotlib.pyplot as plt
def norm(x, dim):
    x = x - x.mean(dim)
    x = x/(x**2).sum(dim)**0.5
    return x  
def da_cor(x, y, dim='stim'):
    x = norm(x, dim)
    y = norm(y, dim)
    r = (x*y).sum(dim)
    return r
    
#%% r2c_vs_r2_cithresh_examples FIG 7

# need to make these two files
cor_dat = pd.read_csv('/Users/dean/Desktop/code/science/modules/sig_cor/data/r2c_ci_cor_dat.cvs',
                      index_col=0)
cor_dat['cilen']  = 1-(cor_dat['ul'] - cor_dat['ll'])
s = xr.open_dataarray('/Users/dean/Desktop/code/science/modules/sig_cor/data/mt_dot_sum_sqrt.nc' )
s.coords['rec'] = s.coords['nms']
#%%
plt.figure(figsize=(3,3))
cor_dat_thresh = cor_dat[(cor_dat['ul']-cor_dat['ll'])<0.5]
ci = np.array([-cor_dat_thresh['ll'].values+cor_dat_thresh['r2c'].values, 
               cor_dat_thresh['ul'].values-cor_dat_thresh['r2c'].values])
plt.errorbar(x=cor_dat_thresh['r2'], y=cor_dat_thresh['r2c'], yerr=ci, fmt='x', c='r');
plt.xlim(0,1.1);plt.ylim(0,1.1);plt.axis('square');plt.grid();
plt.plot([0,1], [0,1], c='k')
plt.xlabel('$\hat{r}^2$', fontsize=14);plt.ylabel('$\hat{r}^2_{ER}$', fontsize=14);
plt.xticks([0,.25,.5,.75,1]);plt.yticks([0,.25,.5,.75,1])

plt.title('MT neuron pairs motion\ntuning curve correlation');
plt.annotate('CI len < 0.25\n n=' + str(len(cor_dat_thresh)),(0.5,0.1), )
plt.tight_layout()
#plt.savefig('/Users/deanpospisil/Desktop/modules/r2c/figs/r2c_vs_r2_cithresh.pdf');
#%%%


hi_r2_hi_snr = (cor_dat['r2'].rank()+
                cor_dat['cilen'].rank()).sort_values().index.values[::-1]

lo_r2_hi_snr = (-cor_dat['r2'].rank()+
                cor_dat['cilen'].rank()).sort_values().index.values[::-1]

i=0
plt.figure(figsize=(8,8))
for ranking in [hi_r2_hi_snr, lo_r2_hi_snr]:
    for nm in ranking[:4]:
       i+=1
       plt.subplot(2, 4, i)
        
       u1, u2 = (s.loc[nm].dropna('trial_tot')**2)/2
       n,m = u1.shape
       plt.errorbar(x=u1.coords['dir'].values, 
                    y=u1.mean('trial_tot'),
                    yerr=u1.std('trial_tot')/(n**0.5))
       
       plt.errorbar(x=u1.coords['dir'].values, 
                    y=u2.mean('trial_tot'),
                    yerr=u2.std('trial_tot')/(n**0.5))
       #plt.ylim(0,110)
       plt.xticks([0,180,360])
       if i==5:
           plt.xlabel('Motion direction (deg)')
           plt.ylabel('spk/s')
       else:
           plt.gca().set_xticklabels([])
           #plt.gca().set_yticklabels([])
       w_r_n, ll, ul, r2c, r2, snr, cilen = cor_dat.loc[nm].values
       plt.title(nm+'\n$r^2_{ER}=$' + str(np.round(r2c, 2)) +
                  ', CI $\in$'+ '[' + str(np.round(ll,2))+', '+str(np.round(ul,2))+']' + 
                 ',\n$r^2=$' + str(np.round(r2, 2)) + 
                 ', SNR=' + str(np.round(snr,2)), fontsize=8)
#       plt.legend(['unit 1', 'unit 2'])
plt.tight_layout()
#plt.savefig('/Users/deanpospisil/Desktop/modules/r2c/figs/r2c_vs_r2_cithresh_examples.pdf');
#%%
from scipy import  stats
cor_dat_thresh = cor_dat[(cor_dat['ul']-cor_dat['ll'])<0.25]
plt.figure(figsize=(4,3))
plt.scatter(cor_dat_thresh['snr'], cor_dat_thresh['r2c'], c='r')
plt.semilogx()
r, p = np.round(stats.spearmanr(cor_dat_thresh['snr'], cor_dat_thresh['r2c']),2)
plt.ylabel('$\hat{r}^2_{ER}$', fontsize=14);
plt.xlabel('SNR')
plt.title('Relation between signal correlation and SNR\nSpearman-r=' + str(r) + ', p='+str(p))
plt.tight_layout()
plt.yticks([0, 0.25, 0.5, 0.75, 1])
plt.gca().set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])
plt.grid()


#%%

n_sims = 200
m_stim = 16
n_trials = 5
n_neuron_pairs = 100
n_row = 7;n_col=3;
plt.figure(figsize=(9,14))
for i, top_pref_theta in enumerate([ np.pi/4, np.pi*2, np.pi/4,]):
    theta = np.linspace(0, 2*np.pi - 2*np.pi/m_stim, m_stim)[np.newaxis, :, 
                       np.newaxis,np.newaxis,np.newaxis]
    pref_theta = np.random.uniform(0, top_pref_theta, 
                                   (n_sims, 1, 1, n_neuron_pairs, 2))
    amp = np.random.uniform(0, 200, (n_sims, 1, 1, n_neuron_pairs, 2))
    tuning_curves = (np.cos(theta + pref_theta) + 1 )*amp/2 + 5
    if i==2:
        sig_noise = np.random.normal(scale=5, size=tuning_curves.shape)
        tuning_curves = sig_noise + tuning_curves
        tuning_curves[tuning_curves<0] = 0
    resp = np.random.poisson(tuning_curves, (n_sims, m_stim, n_trials, 
                                             n_neuron_pairs, 2))
    tc = xr.DataArray(tuning_curves, dims=['sim', 'stim', 'trial' ,'rec', 'pair'], 
                      coords=[range(a) for a in tuning_curves.shape]).squeeze()
    resp = xr.DataArray(resp, dims=['sim', 'stim', 'trial', 'rec', 'pair'],
                        coords=[range(a) for a in resp.shape])

    #
    k=0
    ind = np.ravel_multi_index((k,i), (n_row, n_col));k+=1
    plt.subplot(n_row, n_col, ind+1)
    plt.scatter(np.rad2deg(pref_theta.ravel())[::20], amp.ravel()[::20], s=1)
    plt.ylim(0,220);plt.xlim(-1,361)
    if i ==1:
        plt.title('pref ori. $\in[0,360]$  ')

    elif i==0:
        plt.title('pref ori. $\in[0,45]$ ')
        plt.title('Tuning parameter \n sampling distribution\npref. ori.$\in[0,45]$')

        plt.xlabel('Pref. ori (deg)');
        plt.ylabel('Max-min\ntuning curve amp')
    else:
        plt.title('noisy tuning curves \n pref ori. $\in[0,45]$  ')
    
    #
    ind = np.ravel_multi_index((k,i), (n_row,n_col));k+=1
    plt.subplot(n_row, n_col, ind+1)
    plt.plot(np.rad2deg(theta).squeeze(), tc[0,:,0,0])
    plt.plot(np.rad2deg(theta).squeeze(), tc[0,:,0,1].squeeze())
    plt.ylim(0,200)
    if i==0:
       plt.ylabel('expected spike count')
       plt.xlabel('Motion orientation (deg)')
       plt.title('Example tuning curve pair')
    else:
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])

    
        #
    ind = np.ravel_multi_index((k,i), (n_row,n_col));k+=1
    plt.subplot(n_row, n_col, ind+1)
    r2_true = da_cor(tc.sel(pair=0), tc.sel(pair=1), dim='stim')**2
    plt.hist(r2_true.values.ravel(), bins=20, range=(0,1))
    plt.xlim(-0.1,1.1);plt.xticks([0, 0.5, 1]);
    if i==0:
       plt.ylabel('# neuron pairs')
       plt.xlabel('True $r^2_{ER}$')
       #plt.title('Pairs phase dist \n U(0, 360) deg')
    else:
        plt.gca().set_xticklabels([])
        #plt.gca().set_yticklabels([])
        #plt.title('Pairs phase dist \n U(0, 22.5) deg')

    
    resp = resp.transpose('sim','rec', 'pair', 'trial', 'stim')
    r2c, r2_naive = rc.r2c_n2n(resp.sel(pair=0).values**0.5, 
                            resp.sel(pair=1).values**0.5)
    da_r2c = r2_true.copy(deep=True)
    da_r2c[...] = r2c.squeeze()     
    mu2y, mu2x = rc.mu2_hat_obs_n2n(resp.sel(pair=0).values**0.5, 
                                 resp.sel(pair=1).values**0.5)
    dyn = ((mu2x*mu2y)**0.5).squeeze()/0.25
    
    #
    ind = np.ravel_multi_index((k,i), (n_row, n_col));k+=1
    plt.subplot(n_row, n_col, ind+1)
    plt.scatter(dyn[0], r2_naive[0], s=1)
    plt.grid()
    plt.ylim(-0.1,1.5);
    plt.semilogx()
    plt.xticks([1e1, 1e2, 1e3])
    plt.xlim(1e1,1e3)
    if i==0:
       plt.title('Example experiment\nclassic $r_{{signal}}$')
       plt.xlabel('SNR')
       plt.ylabel('Estimate $\hat{r}^2$')
       plt.annotate('n=' + str(n_neuron_pairs), (2e1, 1.2))
    else:
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
    
    #
    ind = np.ravel_multi_index((k,i), (n_row,n_col));k+=1
    plt.subplot(n_row, n_col, ind+1)
    r_snr_rnaive = [stats.spearmanr(adyn, ar2)[0] 
                    for adyn, ar2 in zip(dyn, r2_naive.squeeze())]
    plt.hist(r_snr_rnaive, range=(-1, 1), bins=20)
    plt.grid()
    plt.xlim(-1, 1);
    plt.xticks([-1,-.5, 0, 0.5, 1])
    if i==0:
       plt.title('$r_s$ all experiments avg. $r_s=$'+ 
                 str(np.round(np.median(r_snr_rnaive), 2)))
       plt.xlabel('$r_s$(SNR, $\hat{r^2}$)')
       plt.ylabel('# experiments (n='+str(n_sims) +')' )
    else:
        plt.gca().set_xticklabels([])
        plt.title(' avg $r_s=$'+ 
                 str(np.round(np.median(r_snr_rnaive), 2)))
        #plt.gca().set_yticklabels([])
    
    #
    ind = np.ravel_multi_index((k,i), (n_row,n_col));k+=1
    plt.subplot(n_row, n_col, ind+1)
    plt.scatter(dyn[0], r2c[0], s=1)
    plt.grid()
    plt.ylim(-0.1,1.5)
    plt.semilogx()    
    plt.xticks([1e1, 1e2, 1e3])
    plt.xlim(1e1,1e3)
    if i==0:
       plt.title('Example experiment\ncorrected $r_{{signal}}$')
       plt.xlabel('SNR')
       plt.ylabel('Estimate $\hat{r}^2_{ER}$')
       plt.annotate('n=' + str(n_neuron_pairs), (2e1, 1.2))
    else:
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        
    #
    ind = np.ravel_multi_index((k,i), (n_row,n_col));k+=1
    plt.subplot(n_row, n_col, ind+1)
    r_snr_r_er = [stats.spearmanr(adyn, ar2)[0] 
                    for adyn, ar2 in zip(dyn, r2c.squeeze())]
    plt.hist(r_snr_r_er, range=(-1, 1), bins=20)
    plt.grid()
    plt.xticks([-1,-.5, 0, 0.5, 1])

    plt.xlim(-1, 1);
    if i==0:
       plt.title('$r_s$ all experiments\navg. $r_s=$' + 
                 str(np.round(np.median(r_snr_r_er), 2)))
       plt.xlabel('$r_s$(SNR, $\hat{r}^2_{ER}$)')
       plt.ylabel('# experiments (n='+str(n_sims) +')' )
    else:
        plt.gca().set_xticklabels([])
        plt.title('avg. $r_s=$' +  
         str(np.round(np.median(r_snr_r_er), 2)))
        #plt.gca().set_yticklabels([])
    
plt.tight_layout()   
#plt.savefig('./figs/snr_r2c_null_sim.pdf')

#%%

load_dir = '/loc6tb/data/responses/v1dotd/'
load_dir = '/Users/deanpospisil/Desktop/modules/r2c/data/v1dotd/'
load_dir = './data/v1dotd/'

fns = os.listdir(load_dir)
fns = [fn for fn in fns if '.nc' in fn ]

v1 = [xr.open_dataarray(load_dir+fn).load() for  fn in fns]
fns = [fn.split('.')[0] for fn in fns]
v1 = xr.concat(v1, 'rec')
v1.coords['rec'] = range(len(fns))

v1.coords['nms'] = ('rec', [fn[:-1] for fn in fns])


v1_s = v1.sel(t=slice(0,2), unit=[0,1])
s = v1_s.sum('t', min_count=1)
s = s**0.5    
ind = ((~s.isnull()).sum('trial_tot')>9).prod('dir').prod('unit').values
s = s[(ind==1)] 
s = s.dropna('rec', how='all').dropna('trial_tot')
#s = s/s.std('trial_tot')
rs1 = np.array(rc.r2c_n2n(s[:, 0, 1::2].values, s[:, 1, ::2].values)).squeeze()[0]
rs2 = np.array(rc.r2c_n2n(s[:, 0, ::2].values, s[:, 1, 1::2].values)).squeeze()[0]
rs = (rs1 + rs2)/2
snr = ((s.mean('trial_tot').var('dir')/
       s.var('trial_tot').mean('dir')).prod('unit')**0.5)
nms = s.coords['nms']
#%%
top_pref_theta = 2*np.pi
m_stim = 8
n_sims = 100
n_neuron_pairs= 81
n_trials=10
theta = np.linspace(0, 2*np.pi - 2*np.pi/m_stim, m_stim)[np.newaxis, :, 
                   np.newaxis,np.newaxis,np.newaxis]
pref_theta = np.random.uniform(0, top_pref_theta, 
                               (n_sims, 1, 1, n_neuron_pairs, 2))
amp = np.random.uniform(0, 100, (n_sims, 1, 1, n_neuron_pairs, 2))
tuning_curves = (np.cos(theta + pref_theta) + 1 )*amp/2 + 5

resp = np.random.poisson(tuning_curves, (n_sims, m_stim, n_trials, 
                                         n_neuron_pairs, 2))
tc = xr.DataArray(tuning_curves, dims=['sim', 'stim', 'trial' ,'rec', 'pair'], 
                  coords=[range(a) for a in tuning_curves.shape]).squeeze()
resp = xr.DataArray(resp, dims=['sim', 'stim', 'trial', 'rec', 'pair'],
                    coords=[range(a) for a in resp.shape])

resp = resp.sel(pair=0, sim=0)
resp = resp.transpose('rec', 'trial', 'stim')
#%%
theta = np.deg2rad(s.coords['dir'].values)[:,np.newaxis]
a = np.concatenate([np.cos(theta), np.sin(theta), 
                    np.ones((len(theta), 1))], -1)

pred_0 = np.dot(a, np.linalg.lstsq(a, 
                                   s.mean('trial_tot').sel(unit=0).values.T,
                                   rcond=None)[0])

pred_1 = np.dot(a, np.linalg.lstsq(a, 
                                   s.mean('trial_tot').sel(unit=1).values.T, 
                                   rcond=None)[0])
pred_null = np.dot(a, np.linalg.lstsq(a, 
                                      resp.mean('trial').values.T, 
                                      rcond=None)[0])

pred_0_fit = np.squeeze([rc.r2c_n2m(a_pred, aunit.values)[0] 
              for a_pred, aunit in zip(pred_0.T, s.sel(unit=0))])

pred_1_fit = np.squeeze([rc.r2c_n2m(a_pred, aunit.values)[0] 
              for a_pred, aunit in zip(pred_1.T, s.sel(unit=1))])

pred_null_fit = np.squeeze([rc.r2c_n2m(a_pred, aunit.values)[0] 
              for a_pred, aunit in zip(pred_null.T, resp)])

snr = ((s.mean('trial_tot').var('dir')/
       s.var('trial_tot').mean('dir')))**0.5

snr_null = (resp.mean('trial').var('stim')/
            resp.var('trial').mean('stim'))**0.5
ind = snr>1
fig = plt.figure(figsize=(4,3))
ax = plt.axes()
ax.plot(snr[:,0], pred_0_fit, 'bo', mfc='none');
ax.plot(snr[:,1], pred_1_fit, 'go', mfc='none');
ax.plot(snr_null, pred_null_fit, 'ko',  mfc='none');

ax.set_xlim(9e-1,1.1e1)
ax.set_ylim(-0.1,1.25)
ax.set_xticks([1, 10])
ax.semilogx()
ax.set_xticks([1, 10])




print(stats.spearmanr(snr[:,1][ind[:,1]],
                np.array(pred_1_fit).squeeze()[ind[:,1]]))
print(stats.spearmanr(snr[:,0][ind[:,0]], 
                np.array(pred_0_fit).squeeze()[ind[:,0]]))

plt.title('MT motion dir. tuning fit to sinusoid model')
plt.xlabel('SNR')
plt.ylabel('$\hat{r}^2_{ER}$')
plt.legend(['unit 1', 'unit 2', 'sinusoid sim.'])
plt.tight_layout()
plt.grid()
#plt.savefig('/home/dean/Desktop/modules/r2c/figs/r2_sin_vs_snr.pdf')
plt.savefig('./figs/r2_sin_vs_snr.pdf')

#%%
ind = 20
inds = np.argsort(pred_0_fit)
m = s.isel(unit=0, rec=inds[ind]).mean('trial_tot')
sd = s.isel(unit=0, rec=inds[ind]).std('trial_tot')
plt.errorbar(x=s.coords['dir'], y=m, yerr=sd/3)
m = s.isel(unit=1, rec=inds[ind]).mean('trial_tot')
sd = s.isel(unit=1, rec=inds[ind]).std('trial_tot')
plt.errorbar(x=s.coords['dir'], y=m, yerr=sd/3)
plt.title(pred_0_fit[inds[ind]])
plt.ylim(0,20)
