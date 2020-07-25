#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 09:48:35 2020

@author: dean
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import os
import xarray as xr
from scipy.stats import spearmanr

def noise_cor_fast(x, y):
    #n,m,sim
    X_ms = x-x.mean(0)
    X_ms_n = X_ms/(X_ms**2.).sum(0)**0.5
    X_ms_n[np.isnan(X_ms_n)] = 0
    
    Y_ms = y-y.mean(0)
    Y_ms_n = Y_ms/(Y_ms**2.).sum(0)**0.5
    
    Y_ms_n[np.isnan(Y_ms_n)] = 0
    
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


def fz(r):
    return 0.5*(np.log((1+r)/(1-r)))

def inv_fz(z):
    return (np.exp(2*z) - 1)/(np.exp(2*z) + 1)

#%%
load_dir = '/loc6tb/data/responses/v1dotd/'
load_dir = '/Users/deanpospisil/Desktop/modules/r2c/data/v1dotd/'
load_dir = '../../data/mt_dotd/'

fns = os.listdir(load_dir)
fns = [fn for fn in fns if '.nc' in fn ]

mt = [xr.open_dataarray(load_dir+fn).load() for  fn in fns]
fns = [fn.split('.')[0] for fn in fns]
mt = xr.concat(mt, 'rec')
mt.coords['rec'] = range(len(fns))

mt.coords['nms'] = ('rec', [fn[:-1] for fn in fns])


mt_s = mt.sel(t=slice(0,2), unit=[0,1])
s = mt_s.sum('t', min_count=1)
s = s**0.5    
ind = ((~s.isnull()).sum('trial_tot')>9).prod('dir').prod('unit').values
s = s[(ind==1)] 
s = s.dropna('rec', how='all').dropna('trial_tot')
#s = s/s.std('trial_tot')

snr = ((s.mean('trial_tot').var('dir')/
       s.var('trial_tot').mean('dir')).prod('unit')**0.5)
nms = s.coords['nms']

#%%
s = s.transpose('unit', 'trial_tot', 'dir', 'rec')

x, y = [s.values[0], s.values[1]]

hat_r_n = noise_cor_fast(x, y)
hat_r_s = sig_cor_fast(x, y)
hat_r_s_split = sig_cor_fast_split(x, y)


#%%
print(np.corrcoef(fz(hat_r_s), fz(hat_r_n)))
print(np.corrcoef(fz(hat_r_s_split), fz(hat_r_n)))



