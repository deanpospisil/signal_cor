#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:25:25 2020

@author: dean
"""

import os
import matplotlib.pyplot as plt
from tqdm import tqdm
main_dir = os.curdir
import numpy as np
import xarray as xr
from itertools import product
import sys


def fz(r):
    return 0.5*(np.log((1+r)/(1-r)))

def inv_fz(z):
    return (np.exp(2*z) - 1)/(np.exp(2*z) + 1)


def sig_cor(x, y):
    #n,m,sim
    X_m = x.mean(0)
    Y_m = y.mean(0)
    
    hat_r_sn  = np.corrcoef(X_m.ravel(), Y_m.ravel())[0,1]
    
    return hat_r_sn

def noise_cor(x, y):
    #n,m,sim
    X_ms = x-x.mean(0)
    Y_ms = y-y.mean(0)
    
    hat_r_n  = np.corrcoef(X_ms.ravel(), Y_ms.ravel())[0,1]
    return hat_r_n

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


def sig_cor_fast_split(x, y):
    #n,m,sim
    mid = int(x.shape[0]/2)
    hat_r_s1 = sig_cor_fast(x[:mid], y[mid:])
    hat_r_s2 = sig_cor_fast(x[mid:], y[:mid])
    hat_r_s = (hat_r_s1 + hat_r_s2)/2
    return hat_r_s

def sig_noise_cor(x, y):
    
    X_ms = x - x.mean(0)
    X_ms_n = X_ms/(X_ms**2).sum(0)**0.5
    
    Y_ms = y - y.mean(0)
    Y_ms_n = Y_ms/(Y_ms**2).sum(0)**0.5
    
    hat_r_sn = (Y_ms_n*X_ms_n).sum(0)
    
    return hat_r_sn

def n2n_pearson_r_sin(r, m):

    angle = np.arccos(r)[np.newaxis, :]  
    s = np.linspace(0, 2*np.pi, int(m))[:, np.newaxis, np.newaxis]
   
    xu = np.cos(s + angle*0)
    yu = np.cos(angle + s)

    yu = (yu/((yu**2.).sum(0)**0.5))
    xu = (xu/((xu**2.).sum(0)**0.5))

   
    return np.array([xu, yu])



def gen_sig_noise_cor_sim_fixed_stim(rho_rs_rn, n_sims, p_cell_pairs, 
                                     sig_n, sig_s, 
                                     snr, n, m,
                                     n_x, n_y, u_n, u_s, split=False):
    
    temp = np.random.multivariate_normal([u_n, u_s], 
                                         [[sig_n**2, sig_n*sig_s*rho_rs_rn],
                                         [sig_n*sig_s*rho_rs_rn, sig_s**2,]],
                              size=[1, 1, p_cell_pairs, n_sims])
    
    a_x = a_y = 5
    z_n = temp[..., 0]
    z_s = temp[..., 1]
    
    r_n = inv_fz(z_n)
    r_s = inv_fz(z_s).squeeze()
    
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
    
    if split:
        hat_r_s = sig_cor_fast_split(X, Y)
    else:    
        hat_r_s = sig_cor_fast(X, Y)
    
    r_rsn = sig_noise_cor(fz(hat_r_s), fz(hat_r_n))
    
    return r_rsn, hat_r_s, hat_r_n, r_s, r_n, Y_s, X_s, Y, X



def gen_a_sig_noise_cor_sim_fixed_stim(r_s, r_n, n_sims, 
                                     snr, n, m,
                                     n_x, n_y,  a_x, a_y, split=False):
    
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
    
    if split:
        hat_r_s = sig_cor_fast_split(X, Y)
    else:    
        hat_r_s = sig_cor_fast(X, Y)
    
    r_rsn = sig_noise_cor(fz(hat_r_s), fz(hat_r_n))
    
    return Y_s, X_s, Y, X


'''
rho_rs_rn = 0.8
n_sims = 100
p_cell_pairs = 50

snr = 1
n = 5
m = 9
n_x = 0.5
n_y = 0.5
a_x = 2
a_y = 2
r_s = np.array([0.5,])
r_n = np.array([0.5,])

Y_s, X_s, Y, X = gen_a_sig_noise_cor_sim_fixed_stim(r_s, r_n, n_sims, 
                                     snr, n, m,
                                     n_x, n_y,  a_x, a_y, split=False)


#%%
n_sims = 100
p_cell_pairs = 50
sig_n = 0.3
sig_s = 0.5
n = 3
n_x=1
n_y=1
u_n=0.1
u_s=0.5
split = False

for rho_rs_rn, snr, m  in zip([0, 0.8], [0.1, 1], [9 , 9]):

    r_rsn, hat_r_s, hat_r_n, r_s, r_n, Y_s, X_s, Y, X = gen_sig_noise_cor_sim_fixed_stim(rho_rs_rn,
                                        n_sims, p_cell_pairs, 
                                         sig_n, sig_s, 
                                         snr, n, m,
                                         n_x, n_y, u_n, u_s, split=False)
    
    r_n = r_n.squeeze()
    Y_da = xr.DataArray(Y, dims=('n', 'm', 'unit', 'sim'),
                        coords=[range(s) for s in Y.shape])
    X_da = xr.DataArray(X, dims=('n', 'm', 'unit', 'sim'),
                        coords=[range(s) for s in Y.shape])
    Y_s_da = xr.DataArray(Y_s, dims=('n', 'm', 'unit', 'sim'),
                        coords=[range(s) for s in Y_s.shape])
    X_s_da = xr.DataArray(X_s, dims=('n', 'm', 'unit', 'sim'),
                        coords=[range(s) for s in Y_s.shape])
    r_s_da = xr.DataArray(r_s, dims=('unit', 'sim'),
                        coords=[range(s) for s in r_s.shape])
    r_n_da = xr.DataArray(r_n, dims=('unit', 'sim'),
                        coords=[range(s) for s in r_n.shape])
    hat_r_s_da = xr.DataArray(hat_r_s, dims=('unit', 'sim'),
                        coords=[range(s) for s in hat_r_s.shape])
    hat_r_n_da = xr.DataArray(hat_r_n, dims=('unit', 'sim'),
                        coords=[range(s) for s in hat_r_n.shape])
    
    r_rsn_da = xr.DataArray(r_rsn, dims=('sim'),
                        coords=[range(s) for s in r_rsn.shape])
    
    d = xr.Dataset({'Y':Y_da, 'X':X_da, 
                    'Y_s':Y_s_da, 'X_s':X_s_da,
                    'r_s':r_s_da, 'r_n':r_n_da,
                    'hat_r_s':hat_r_s_da, 'hat_r_n':hat_r_n_da,
                    'r_rsn':r_rsn_da})
    d.attrs = {'snr':snr, 'u_n':u_n, 'u_s':u_s, 'n_x':n_x, 
               'm':m, 'n':n, 'rho_rs_rn':rho_rs_rn, 'p_cell_pairs':p_cell_pairs}
    
    nm = 'example_sim_fxd_stim_rn='+ str(rho_rs_rn)+'.nc'
    if os.path.exists(nm):
      os.remove(nm)
    d.to_netcdf(nm)


#%%
n_sims = 5
p_cell_pair = 50


#fixed
# WYETH: match what people think is plausible.
# just match what you
# why is there this small gap.
# signal corr between 0-1
# noise corr between 0 and 0.3
sig_s = 0.5
u_s = 0.5
sig_n = 0.1
u_n = 0.15
rho_rs_rns = [0, 0.25, 0.5, 0.75]


ns = [10, ]
ms = [10, 100, 1000,]
p_cell_pairs = [p_cell_pair,]
snrs = [0.1,]

a = np.array(list(product(*[ns, ms, rho_rs_rns, snrs, p_cell_pairs])))
da = xr.DataArray(np.zeros((len(ns), 
                              len(ms), 
                              len(rho_rs_rns),
                              len(snrs),
                              len(p_cell_pairs),
                              n_sims)), 
             dims=['n', 'm', 'rho', 'snr', 'p_cell_pairs', 'sim'],
             coords =[ns, ms, rho_rs_rns,  snrs, p_cell_pairs, range(n_sims)])
for p in tqdm(a):
    a_n, a_m, a_rho, a_snr, a_p_cell_pairs = p
    r_rsn = gen_sig_noise_cor_sim_fixed_stim(a_rho, n_sims, int(a_p_cell_pairs),
                                  sig_n, sig_s, a_snr, 
                                  int(a_n), int(a_m), n_x=1, n_y=1,
                                  u_n=u_n, u_s=u_s)[0]
    da.loc[a_n, a_m, a_rho, a_snr, a_p_cell_pairs, a_split] = r_rsn


if os.path.exists("rho_sn_vs_m_snr_low_fxd_stim.nc"):
  os.remove("rho_sn_vs_m_snr_low_fxd_stim.nc")

da.to_netcdf('rho_sn_vs_m_snr_low_fxd_stim.nc')




snrs = [100.,]
a = np.array(list(product(*[ns, ms, rho_rs_rns, snrs, p_cell_pairs])))
da = xr.DataArray(np.zeros((len(ns), 
                              len(ms), 
                              len(rho_rs_rns),
                              len(snrs),
                              len(p_cell_pairs),
                              n_sims)), 
             dims=['n', 'm', 'rho', 'snr', 'p_cell_pairs', 'sim'],
             coords =[ns, ms, rho_rs_rns,  snrs, p_cell_pairs, range(n_sims)])
for p in tqdm(a):
    a_n, a_m, a_rho, a_snr, a_p_cell_pairs = p
    r_rsn = gen_sig_noise_cor_sim_fixed_stim(a_rho, n_sims, int(a_p_cell_pairs),
                                  sig_n, sig_s, a_snr, 
                                  int(a_n), int(a_m), n_x=1, n_y=1, u_n=u_n, u_s=u_s)[0]
    da.loc[a_n, a_m, a_rho, a_snr, a_p_cell_pairs, a_split] = r_rsn


if os.path.exists("rho_sn_vs_m_snr_hi_fxd_stim.nc"):
  os.remove("rho_sn_vs_m_snr_hi_fxd_stim.nc")

da.to_netcdf('rho_sn_vs_m_snr_hi_fxd_stim.nc')


ns = [10, ]
ms = [100,]
snrs = [0.1, 1, 10., 100.]
a = np.array(list(product(*[ns, ms, rho_rs_rns, snrs, p_cell_pairs])))

da = xr.DataArray(np.zeros((len(ns), 
                              len(ms), 
                              len(rho_rs_rns),
                              len(snrs),
                              len(p_cell_pairs),
                              n_sims)), 
             dims=['n', 'm', 'rho', 'snr', 'p_cell_pairs', 'sim'],
             coords =[ns, ms, rho_rs_rns,  snrs, p_cell_pairs, range(n_sims)])

for p in tqdm(a):
    a_n, a_m, a_rho, a_snr, a_p_cell_pairs = p
    r_rsn = gen_sig_noise_cor_sim_fixed_stim(a_rho, n_sims, int(a_p_cell_pairs),
                                  sig_n, sig_s, a_snr, 
                                  int(a_n), int(a_m), n_x=1, n_y=1, u_n=u_n, u_s=u_s)[0]
    da.loc[a_n, a_m, a_rho, a_snr, a_p_cell_pairs, a_split] = r_rsn


if os.path.exists("rho_sn_vs_SNR_fxd_stim.nc"):
  os.remove("rho_sn_vs_SNR_fxd_stim.nc")

da.to_netcdf('rho_sn_vs_SNR_fxd_stim.nc')

ns = [4, 8, 16, 32, 64]
snrs = [0.1,]
a = np.array(list(product(*[ns, ms, rho_rs_rns, snrs, p_cell_pairs])))
da = xr.DataArray(np.zeros((len(ns), 
                              len(ms), 
                              len(rho_rs_rns),
                              len(snrs),
                              len(p_cell_pairs),
                              n_sims)), 
             dims=['n', 'm', 'rho', 'snr', 'p_cell_pairs', 'sim'],
             coords =[ns, ms, rho_rs_rns,  snrs, p_cell_pairs, range(n_sims)])
for p in tqdm(a):
    a_n, a_m, a_rho, a_snr, a_p_cell_pairs = p
    r_rsn = gen_sig_noise_cor_sim_fixed_stim(a_rho, n_sims, int(a_p_cell_pairs),
                                  sig_n, sig_s, a_snr, 
                                  int(a_n), int(a_m), n_x=1, n_y=1, 
                                  u_n=u_n, u_s=u_s)[0]
    da.loc[a_n, a_m, a_rho, a_snr, a_p_cell_pairs, a_split] = r_rsn

if os.path.exists("rho_sn_vs_n_fxd_stim.nc"):
  os.remove("rho_sn_vs_n_fxd_stim.nc")

da.to_netcdf('rho_sn_vs_n_fxd_stim.nc')


ns = [10, ]
ms = [100,]
snrs = [0.1, 1, 10., 100]
p_cell_pairs = [p_cell_pair,]

a = np.array(list(product(*[ns, ms, rho_rs_rns, snrs, p_cell_pairs])))

da = xr.DataArray(np.zeros((len(ns), 
                              len(ms), 
                              len(rho_rs_rns),
                              len(snrs),
                              len(p_cell_pairs),
                              n_sims)), 
             dims=['n', 'm', 'rho', 'snr', 'p_cell_pairs', 'sim'],
             coords =[ns, ms, rho_rs_rns,  snrs, p_cell_pairs, range(n_sims)])

for p in tqdm(a):
    a_n, a_m, a_rho, a_snr, a_p_cell_pairs = p
    r_rsn = gen_sig_noise_cor_sim_fixed_stim(a_rho, n_sims, int(a_p_cell_pairs),
                                  sig_n, sig_s, a_snr, 
                                  int(a_n), int(a_m),
                                  n_x=1, n_y=1, u_n=u_n, u_s=u_s, split=True)[0]
    da.loc[a_n, a_m, a_rho, a_snr, a_p_cell_pairs, a_split] = r_rsn


if os.path.exists("rho_sn_vs_SNR_split_fxd_stim.nc"):
  os.remove("rho_sn_vs_SNR_split_fxd_stim.nc")

da.to_netcdf('rho_sn_vs_SNR_split_fxd_stim.nc')



#%%
sys.path.append('/Users/dean/Desktop/code/science/modules/r2c')
import r2c_common as r2c
from common_cor_sim import gen_a_sig_noise_cor_sim_fixed_stim
os.chdir(main_dir)
n_sims = 3000
ns = [8, ]
ms = [500,]
snrs = [0.1, 0.5, 1, 10]
r_ns = [0, 0.25, 0.75,]
r_ss = np.array([0, 0.25, 0.5, 0.75, 1])**0.5

a = np.array(list(product(*[ns, ms,  snrs, r_ns, r_ss])))

da = xr.DataArray(np.zeros((len(ns), 
                              len(ms), 
                              len(snrs),
                              len(r_ss),
                              len(r_ns),
                              n_sims,
                              4)), 
             dims=['n', 'm', 'snr', 'r_s', 'r_n', 'sim', 'est'],
             coords =[ns,  ms,   snrs, r_ss, r_ns, range(n_sims), ['r2', 'r2_split', 'r2_er', 'r2_er_split']])

da[...] = np.nan
for p in tqdm(a):
    n, m,  snr, r_n, r_s = p
    Y_s, X_s, Y, X, hat_r_n, hat_r_s = gen_a_sig_noise_cor_sim_fixed_stim(r_s, r_n, int(n_sims), 
                                     snr, int(n), int(m),
                                     n_x=1, n_y=1,  a_x=0, a_y=0)
    
    r2 = (hat_r_s**2).squeeze()
    r2_split = np.squeeze(sig_cor_fast_split(X, Y)**2)
    X = np.squeeze(X)
    X = np.transpose(X, [2, 0, 1])
    Y = np.squeeze(Y)
    Y = np.transpose(Y, [2, 0, 1])
    r2_er = np.squeeze(r2c.r2c_n2n(X, Y)[0])
    
    r2_er_split = (np.squeeze(r2c.r2c_n2n(X[:, 1::2], Y[:, ::2])[0]) + 
                   np.squeeze(r2c.r2c_n2n(X[:, ::2], Y[:, 1::2])[0]))/2
    
    da.loc[n, m,  snr, r_s, r_n, :, 'r2'] = r2
    da.loc[n, m,  snr, r_s, r_n, :, 'r2_split'] = r2_split
    da.loc[n, m,  snr, r_s, r_n, :, 'r2_er'] = r2_er
    da.loc[n, m,  snr, r_s, r_n, :, 'r2_er_split'] = r2_er_split

nm = 'sim_r2s_est_comp.nc'
if os.path.exists(nm):
  os.remove(nm)
da.to_netcdf(nm)  
'''
#%%
n_sims = 500
p_cell_pair = 50


#fixed
# WYETH: match what people think is plausible.
# just match what you
# why is there this small gap.
# signal corr between 0-1
# noise corr between 0 and 0.3
sig_s = 0.5
u_s = 0.5
sig_n = 0.1
u_n = 0.15
rho_rs_rns = [0, 0.25, 0.5, 0.75]


ns = [10, ]
ms = [10, 100, 1000,]
p_cell_pairs = [p_cell_pair,]
snrs = [0.1,]
split = [False,True]

a = np.array(list(product(*[ns, ms, rho_rs_rns, snrs, p_cell_pairs, split])))
da = xr.DataArray(np.zeros((len(ns), 
                              len(ms), 
                              len(rho_rs_rns),
                              len(snrs),
                              len(p_cell_pairs),
                              len(split),
                              n_sims)), 
             dims=['n', 'm', 'rho', 'snr', 'p_cell_pairs', 'split', 'sim'],
             coords =[ns, ms, rho_rs_rns,  snrs, p_cell_pairs, split, range(n_sims)])
for p in tqdm(a):
    a_n, a_m, a_rho, a_snr, a_p_cell_pairs, a_split = p
    r_rsn = gen_sig_noise_cor_sim_fixed_stim(a_rho, n_sims, int(a_p_cell_pairs),
                                  sig_n, sig_s, a_snr, 
                                  int(a_n), int(a_m), n_x=1, n_y=1,
                                  u_n=u_n, u_s=u_s, split=a_split)[0]
    da.loc[a_n, a_m, a_rho, a_snr, a_p_cell_pairs, a_split] = r_rsn


if os.path.exists("rho_sn_vs_m_snr_low_fxd_stim.nc"):
  os.remove("rho_sn_vs_m_snr_low_fxd_stim.nc")

da.to_netcdf('rho_sn_vs_m_snr_low_fxd_stim.nc')



snrs = [100.,]
a = np.array(list(product(*[ns, ms, rho_rs_rns, snrs, p_cell_pairs, split])))
da = xr.DataArray(np.zeros((len(ns), 
                              len(ms), 
                              len(rho_rs_rns),
                              len(snrs),
                              len(p_cell_pairs),
                              len(split),
                              n_sims)), 
             dims=['n', 'm', 'rho', 'snr', 'p_cell_pairs', 'split', 'sim'],
             coords =[ns, ms, rho_rs_rns,  snrs, p_cell_pairs, split, range(n_sims)])
for p in tqdm(a):
    a_n, a_m, a_rho, a_snr, a_p_cell_pairs, a_split = p
    r_rsn = gen_sig_noise_cor_sim_fixed_stim(a_rho, n_sims, int(a_p_cell_pairs),
                                  sig_n, sig_s, a_snr, 
                                  int(a_n), int(a_m), n_x=1, n_y=1,
                                  u_n=u_n, u_s=u_s, split=a_split)[0]
    da.loc[a_n, a_m, a_rho, a_snr, a_p_cell_pairs, a_split] = r_rsn


if os.path.exists("rho_sn_vs_m_snr_hi_fxd_stim.nc"):
  os.remove("rho_sn_vs_m_snr_hi_fxd_stim.nc")

da.to_netcdf('rho_sn_vs_m_snr_hi_fxd_stim.nc')


ns = [10, ]
ms = [100,]
snrs = [0.1, 1, 10., 100.]
a = np.array(list(product(*[ns, ms, rho_rs_rns, snrs, p_cell_pairs, split])))
da = xr.DataArray(np.zeros((len(ns), 
                              len(ms), 
                              len(rho_rs_rns),
                              len(snrs),
                              len(p_cell_pairs),
                              len(split),
                              n_sims)), 
             dims=['n', 'm', 'rho', 'snr', 'p_cell_pairs', 'split', 'sim'],
             coords =[ns, ms, rho_rs_rns,  snrs, p_cell_pairs, split, range(n_sims)])
for p in tqdm(a):
    a_n, a_m, a_rho, a_snr, a_p_cell_pairs, a_split = p
    r_rsn = gen_sig_noise_cor_sim_fixed_stim(a_rho, n_sims, int(a_p_cell_pairs),
                                  sig_n, sig_s, a_snr, 
                                  int(a_n), int(a_m), n_x=1, n_y=1,
                                  u_n=u_n, u_s=u_s, split=a_split)[0]
    da.loc[a_n, a_m, a_rho, a_snr, a_p_cell_pairs, a_split] = r_rsn



if os.path.exists("rho_sn_vs_SNR_fxd_stim.nc"):
  os.remove("rho_sn_vs_SNR_fxd_stim.nc")

da.to_netcdf('rho_sn_vs_SNR_fxd_stim.nc')

ns = [4, 8, 16, 32, 64]
snrs = [0.1,]
a = np.array(list(product(*[ns, ms, rho_rs_rns, snrs, p_cell_pairs, split])))
da = xr.DataArray(np.zeros((len(ns), 
                              len(ms), 
                              len(rho_rs_rns),
                              len(snrs),
                              len(p_cell_pairs),
                              len(split),
                              n_sims)), 
             dims=['n', 'm', 'rho', 'snr', 'p_cell_pairs', 'split', 'sim'],
             coords =[ns, ms, rho_rs_rns,  snrs, p_cell_pairs, split, range(n_sims)])
for p in tqdm(a):
    a_n, a_m, a_rho, a_snr, a_p_cell_pairs, a_split = p
    r_rsn = gen_sig_noise_cor_sim_fixed_stim(a_rho, n_sims, int(a_p_cell_pairs),
                                  sig_n, sig_s, a_snr, 
                                  int(a_n), int(a_m), n_x=1, n_y=1,
                                  u_n=u_n, u_s=u_s, split=a_split)[0]
    da.loc[a_n, a_m, a_rho, a_snr, a_p_cell_pairs, a_split] = r_rsn



if os.path.exists("rho_sn_vs_n_fxd_stim.nc"):
  os.remove("rho_sn_vs_n_fxd_stim.nc")

da.to_netcdf('rho_sn_vs_n_fxd_stim.nc')

#instead just use rho_sn_vs_SNR_fxd_stim
# ns = [10, ]
# ms = [100,]
# snrs = [0.1, 1, 10., 100]
# p_cell_pairs = [p_cell_pair,]

# a = np.array(list(product(*[ns, ms, rho_rs_rns, snrs, p_cell_pairs, split])))
# da = xr.DataArray(np.zeros((len(ns), 
#                               len(ms), 
#                               len(rho_rs_rns),
#                               len(snrs),
#                               len(p_cell_pairs),
#                               len(split),
#                               n_sims)), 
#              dims=['n', 'm', 'rho', 'snr', 'p_cell_pairs', 'split', 'sim'],
#              coords =[ns, ms, rho_rs_rns,  snrs, p_cell_pairs, split, range(n_sims)])
# for p in tqdm(a):
#     a_n, a_m, a_rho, a_snr, a_p_cell_pairs, a_split = p
#     r_rsn = gen_sig_noise_cor_sim_fixed_stim(a_rho, n_sims, int(a_p_cell_pairs),
#                                   sig_n, sig_s, a_snr, 
#                                   int(a_n), int(a_m), n_x=1, n_y=1,
#                                   u_n=u_n, u_s=u_s, split=a_split)[0]
#     da.loc[a_n, a_m, a_rho, a_snr, a_p_cell_pairs, a_split] = r_rsn



# if os.path.exists("rho_sn_vs_SNR_split_fxd_stim.nc"):
#   os.remove("rho_sn_vs_SNR_split_fxd_stim.nc")

# da.to_netcdf('rho_sn_vs_SNR_split_fxd_stim.nc')
