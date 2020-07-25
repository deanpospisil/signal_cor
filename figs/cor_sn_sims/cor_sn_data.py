#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:29:00 2020

@author: dean
"""



import matplotlib.pyplot as plt

import numpy as np
import xarray as xr
from itertools import product
import os

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



def gen_sig_noise_cor_sim(rho_rs_rn, n_sims, p_cell_pairs, sig_n, sig_s, 
                          snr, n, m,
                          n_x, n_y, u_n, u_s, split=False):
    
    #np.random.seed(1)
    temp = np.random.multivariate_normal([u_n, u_s], 
                                         [[sig_n**2, sig_n*sig_s*rho_rs_rn],
                                         [sig_n*sig_s*rho_rs_rn, sig_s**2,]],
                              size=[1, 1, p_cell_pairs, n_sims])
    
    a_x = a_y = 1
    z_n = temp[..., 0]
    z_s = temp[..., 1]
    
    r_n = inv_fz(z_n)
    r_s = inv_fz(z_s)
    
    #fixed
    n_y = np.sign(r_n)*n_y
    
    #derived
    d_x = (snr*np.abs(n_x*n_y))**0.5
    d_y = d_x.copy()
    d_y = np.sign(r_s)*d_y
    
    
    S = np.random.normal(0, 1, size=(1, m, p_cell_pairs, n_sims))
    R_x = np.random.normal(0, 1, size=(1, m, p_cell_pairs, n_sims))
    R_y = np.random.normal(0, 1, size=(1, m, p_cell_pairs, n_sims))
    
    C = np.random.normal(0, 1, size=(n, m, p_cell_pairs, n_sims))
    N_x = np.random.normal(0, 1, size=(n, m, p_cell_pairs, n_sims))
    N_y = np.random.normal(0, 1, size=(n, m, p_cell_pairs, n_sims))
    
    X_s =  a_x + d_x*((abs(r_s)**0.5)*S + ((1-abs(r_s))**0.5)*R_x)
    X_n =        n_x*((abs(r_n)**0.5)*C + ((1-abs(r_n))**0.5)*N_x)
    X = X_s + X_n
    
    Y_s =  a_y + d_y*((abs(r_s)**0.5)*S + ((1-abs(r_s))**0.5)*R_y)
    Y_n =        n_y*((abs(r_n)**0.5)*C + ((1-abs(r_n))**0.5)*N_y)
    Y = Y_s + Y_n
    
    hat_r_n = noise_cor_fast(X, Y)
    
    if split:
        hat_r_s = sig_cor_fast_split(X, Y)
    else:    
        hat_r_s = sig_cor_fast(X, Y)

    r_rsn = sig_noise_cor(fz(hat_r_s), fz(hat_r_n))
    
    return r_rsn, hat_r_s, hat_r_n, r_s, r_n



n_sims = 100
p_cell_pair = 50


#fixed
# WYETH: match what people think is plausible.
# just match what you
# why is there this small gap.
# signal corr between 0-1
# noise corr between 0 and 0.3
sig_s = 0.5
sig_n = 0.1
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
for p in a:
    a_n, a_m, a_rho, a_snr, a_p_cell_pairs = p
    r_rsn = gen_sig_noise_cor_sim(a_rho, n_sims, int(a_p_cell_pairs),
                                  sig_n, sig_s, a_snr, 
                                  int(a_n), int(a_m), n_x=1, n_y=1, u_n=0.1, u_s=0.5)[0]
    da.loc[a_n, a_m, a_rho, a_snr, a_p_cell_pairs, :] = r_rsn


if os.path.exists("rho_sn_vs_m_snr_low.nc"):
  os.remove("rho_sn_vs_m_snr_low.nc")

da.to_netcdf('rho_sn_vs_m_snr_low.nc')




snrs = [1000.,]
a = np.array(list(product(*[ns, ms, rho_rs_rns, snrs, p_cell_pairs])))
da = xr.DataArray(np.zeros((len(ns), 
                              len(ms), 
                              len(rho_rs_rns),
                              len(snrs),
                              len(p_cell_pairs),
                              n_sims)), 
             dims=['n', 'm', 'rho', 'snr', 'p_cell_pairs', 'sim'],
             coords =[ns, ms, rho_rs_rns,  snrs, p_cell_pairs, range(n_sims)])
for p in a:
    a_n, a_m, a_rho, a_snr, a_p_cell_pairs = p
    r_rsn = gen_sig_noise_cor_sim(a_rho, n_sims, int(a_p_cell_pairs),
                                  sig_n, sig_s, a_snr, 
                                  int(a_n), int(a_m), n_x=1, n_y=1, u_n=0, u_s=0)[0]
    da.loc[a_n, a_m, a_rho, a_snr, a_p_cell_pairs, :] = r_rsn


if os.path.exists("rho_sn_vs_m_snr_hi.nc"):
  os.remove("rho_sn_vs_m_snr_hi.nc")

da.to_netcdf('rho_sn_vs_m_snr_hi.nc')


ns = [10, ]
ms = [1000,]
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

for p in a:
    a_n, a_m, a_rho, a_snr, a_p_cell_pairs = p
    r_rsn = gen_sig_noise_cor_sim(a_rho, n_sims, int(a_p_cell_pairs),
                                  sig_n, sig_s, a_snr, 
                                  int(a_n), int(a_m), n_x=1, n_y=1, u_n=0, u_s=0)[0]
    da.loc[a_n, a_m, a_rho, a_snr, a_p_cell_pairs, :] = r_rsn


if os.path.exists("rho_sn_vs_SNR.nc"):
  os.remove("rho_sn_vs_SNR.nc")

da.to_netcdf('rho_sn_vs_SNR.nc')

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
for p in a:
    a_n, a_m, a_rho, a_snr, a_p_cell_pairs = p
    r_rsn = gen_sig_noise_cor_sim(a_rho, n_sims, int(a_p_cell_pairs),
                                  sig_n, sig_s, a_snr, 
                                  int(a_n), int(a_m), n_x=1, n_y=1, u_n=0, u_s=0)[0]
    da.loc[a_n, a_m, a_rho, a_snr, a_p_cell_pairs, :] = r_rsn

if os.path.exists("rho_sn_vs_n.nc"):
  os.remove("rho_sn_vs_n.nc")

da.to_netcdf('rho_sn_vs_n.nc')


ns = [10, ]
ms = [1000,]
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

for p in a:
    a_n, a_m, a_rho, a_snr, a_p_cell_pairs = p
    r_rsn = gen_sig_noise_cor_sim(a_rho, n_sims, int(a_p_cell_pairs),
                                  sig_n, sig_s, a_snr, 
                                  int(a_n), int(a_m),
                                  n_x=1, n_y=1, u_n=0, u_s=0, split=True)[0]
    da.loc[a_n, a_m, a_rho, a_snr, a_p_cell_pairs, :] = r_rsn



if os.path.exists("rho_sn_vs_SNR_split.nc"):
  os.remove("rho_sn_vs_SNR_split.nc")

da.to_netcdf('rho_sn_vs_SNR_split.nc')






