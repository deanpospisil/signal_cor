#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:27:53 2019

@author: dean
"""
import r2c_common as rc
import numpy as np
import xarray as xr
import pandas as pd
import os
from scipy import stats
def rect(x):
    if x<0:
        return 0
    return x
def mod_p(m,n,snr):
    c = stats.f(n*m-1, m*(n-1)).ppf(0.99)
    p = stats.ncf(n*m-1, m*(n-1), nc=(n*m)*(snr)**2).cdf(c)
    return p
#%% neuron to model simulations
n_exps = 5000
sig2=0.25;
m=362;n=4;


#max_spike = np.array([0.25, 2 ])
#mu2ys = m*max_spike/4
snrs = np.array([0.1, 0.25, 1])#pulling typical snr values
#snrs = np.array([0.01, 0.05, 1])#pulling typical snr values

ncps = snrs*362*4#choose a desire ncp
mu2ys = ncps*sig2/n#adjust mu2y and consequently snr to match this ncp
snrs_new = ((mu2ys/m))/(sig2)#

mu2x = mu2ys[-1]
r2sims = np.linspace(0,1,5)
ql, qh = [0.05,0.95]

ncp_of_n2m_sims = m*n*((mu2ys/m)/sig2)
#n sims
#Er2, Er2c, ql, qh
_ = np.zeros([6, len(mu2ys), len(r2sims)])
da = xr.DataArray(_, dims=['measure', 'mu2ys', 'r2s'],
               coords=[['r2', 'r2c', 'r2_ql', 'r2_qh', 'r2c_ql', 'r2c_qh'],
                       mu2ys, r2sims])
da.attrs = {'n':n, 'm':m, 'mu2x':mu2x, 'sig2':sig2, 'ql':ql, 'qh':qh, 
            'n_exps':n_exps}

for k, mu2y in enumerate(mu2ys):
    r2cs = []
    r2s = []   
    for r2 in r2sims:
        theta = [r2, sig2, mu2y, mu2y, m, n]
        x,y = rc.pds_n2m_r2c(theta, n_exps, ddof=1)
        x = x.squeeze()[np.newaxis,np.newaxis]
        r2c, r2 = rc.r2c_n2m(x, y)    
        r2cs.append(r2c.squeeze())
        r2s.append(r2.squeeze())
    r2cs = np.array(r2cs)
    r2s = np.array(r2s)
    
    da.loc['r2', mu2y] = r2s.mean(1)
    da.loc['r2c', mu2y] = r2cs.mean(1)
        
    qs = np.array([np.quantile(r2cs[i], [.05, .95]) for i in range(len(r2s))])
    qs = np.abs(qs - r2cs.mean(1, keepdims=True))
    da.loc[['r2c_ql', 'r2c_qh'], mu2y] = qs.T

    qs = np.array([np.quantile(r2s[i], [.05, .95]) for i in range(len(r2s))])
    qs = np.abs(qs - r2s.mean(1, keepdims=True))
    da.loc[['r2_ql', 'r2_qh'], mu2y] = qs.T
os.remove('./figs/fig_data/n2m_sim.nc')    
da.to_netcdf('./figs/fig_data/n2m_sim.nc')
df = xr.open_dataarray('./figs/fig_data/n2m_sim.nc').load()


#%% neuron to neuron sims

n_exps = 5000
sig2=0.25;
m = 362;n=4;
r2sims = np.linspace(0,1,5)
ql, qh = [0.05,0.95]
#n sims
#Er2, Er2c, ql, qh
_ = np.zeros([6, len(mu2ys), len(r2sims)])
da = xr.DataArray(_, dims=['measure', 'mu2ys', 'r2s'],
               coords=[['r2', 'r2c', 'r2_ql', 'r2_qh', 'r2c_ql', 'r2c_qh'],
                       mu2ys, r2sims])
da.attrs = {'n':n, 'm':m, 'mu2x':mu2x, 'sig2':sig2, 'ql':ql, 'qh':qh, 
            'n_exps':n_exps}

for k, mu2y in enumerate(mu2ys):
    r2cs = []
    r2s = []   
    for r2 in r2sims:
        theta = [r2, sig2, mu2y, mu2y, m, n]
        x,y = rc.pds_n2n_r2c(theta, n_exps, ddof=1)
        x = x.squeeze()[np.newaxis,np.newaxis]
        r2c, r2 = rc.r2c_n2n(x, y)
        
        r2cs.append(r2c.squeeze())
        r2s.append(r2.squeeze())
    r2cs = np.array(r2cs)
    r2s = np.array(r2s)
    #plt.hist(r2cs[-1])
    da.loc['r2', mu2y] = r2s.mean(1)
    da.loc['r2c', mu2y] = r2cs.mean(1)
        
    qs = np.array([np.quantile(r2cs[i], [ql, qh]) for i in range(len(r2s))])
    qs = np.abs(qs - r2cs.mean(1, keepdims=True))
    da.loc[['r2c_ql', 'r2c_qh'], mu2y] = qs.T

    qs = np.array([np.quantile(r2s[i], [ql, qh]) for i in range(len(r2s))])
    qs = np.abs(qs - r2s.mean(1, keepdims=True))
    da.loc[['r2_ql', 'r2_qh'], mu2y] = qs.T
os.remove('./figs/fig_data/n2n_sim.nc')
da.to_netcdf('./figs/fig_data/n2n_sim.nc')


df = xr.open_dataarray('./figs/fig_data/n2n_sim.nc')
    
    
#%%
n = 20
m = 121
snrs = np.linspace(0.01, 1, 20)#pulling typical snr values
ncps = snrs*362*4#choose a desire ncp
mu2ys = ncps*sig2/n#adjust mu2y and consequently snr to match this ncp
snrs_new = ((mu2ys/m))/(sig2)#


n_exps = 10000
r2cs = []

for mu2y in mu2ys :
    print(mu2y)
    r2=0.1
    theta = [r2, sig2, mu2y, mu2y, m, n]
    x,y = rc.pds_n2n_r2c(theta, n_exps, ddof=1)
    x = x.squeeze()[np.newaxis,np.newaxis]
    r2c, r2 = rc.r2c_n2n(x, y)
    r2cs.append(r2c)

r2cs = np.array(r2cs).squeeze()    
#%%
plt.plot(snrs_new, np.median(r2cs,1))
plt.semilogx()
#plt.xlim(1e-2,10)
#plt.xlim(0, 0.15)
plt.xlabel('SNR')
plt.ylabel(r'$\hat{r}^2_{ER}$')
plt.title('True r^2_{ER}=0.1')
#%% calc all
import r2c_common as rc

sig2=0.25;
n_exps = 500
r2sims = np.linspace(0, 1, 5)
n = 4;m=362


res = []
ms = np.array([25, 50, 100, 150, 200])
ms = np.array([362,])

ncps = ncp_of_n2m_sims
mu2ys = (np.array(ncps)/(n))*sig2
_ = np.zeros([9, len(ncps), len(r2sims), len(ms), (n_exps)])

da = xr.DataArray(_, dims=['measure',  'mu2ys', 'r2s', 'm', 'exps',],
               coords=[['r2', 
                        'H&C_my_deriv',
                        'P&C',  
                        'H&C_their_deriv', 
                        'Y&D', 
                        'H&T', 
                        'S&L', 
                        'S&S',
                        'Zyl'], 
                       mu2ys, r2sims, ms, range(n_exps)])
da.attrs = {'n':n, 'm':m, 'mu2x':mu2x, 'sig2':sig2,  
            'n_exps':n_exps}
for k, mu2y in enumerate(mu2ys):
    r2cs = []
    r2s = []   
    for l, m in enumerate(ms):
        for j, r2 in enumerate(r2sims):
            theta = [r2, sig2, mu2y, mu2y, m, n]
            [x, y] = rc.pds_n2m_r2c(theta, n_exps, ddof=1)
            res = []
            for i in range(y.shape[0]):
                a_y = y[i]
                mod = np.zeros((len(x), 2))
                mod[:,0] = 1
                mod[:, 1] = x.squeeze()
                beta = np.linalg.lstsq(mod, a_y.mean(0), rcond=-1)[0]
                y_hat = np.dot(beta[np.newaxis], mod.T).squeeze()
                
                r2c, r2 = rc.r2c_n2m(x.T, a_y)  
                r2_pc = rc.r2_SE_corrected(x.squeeze(), a_y)
                r2_upsilon = rc.upsilon(y_hat, a_y)
                r2_hsu = rc.cc_norm_split(x.squeeze(), a_y)**2
                r2_yd = rc.r2_SB_normed(x.squeeze(), a_y)
                r2_sl = rc.normalized_spe(y_hat, a_y)
                r2_sc = rc.cc_norm(x.squeeze(), a_y)**2
                r2_zyl = rc.cc_norm_bs(x.squeeze(), a_y)**2
                res.append([np.double(r2.squeeze()), 
                            np.double(r2c.squeeze()), 
                            r2_pc, 
                            r2_upsilon, 
                            r2_yd, 
                            r2_hsu, 
                            r2_sl, 
                            r2_sc,
                            r2_zyl])
                
            res = np.array(res) 
            da[:,k,j, l, :] = res.T

import os            
os.remove('./figs/fig_data/alt_n2m_meth_sim.nc')
da.to_netcdf('./figs/fig_data/alt_n2m_meth_sim.nc')

p = da.mean('exps')[:,0,-1].to_pandas()

#%%
v4 = xr.open_dataset('./data/apc370_with_trials.nc')['resp']
try:
    v4 = v4.rename({'trials':'trial'})
    v4 = v4.rename({'shapes':'stim'})
except:
    print('')
v4.coords['unit'] = range(109)
ys = []
for cell in range(109):
    y = v4.sel(unit=cell).dropna('trial')
    if (y.shape[1])==4:
        ys.append(y)
temp = xr.concat(ys, 'unit')
ys = temp**0.5

x = ys[:,:,::2]
y = ys[:,:,1::2] 
m = x.shape[1]
n = x.shape[-1]
       

sig2 = ys.var('trial').mean('stim')
x_ms = (x.mean('trial')-x.mean(('trial', 'stim')))
mu2 = (x_ms**2).sum('stim') - ((m-1)*sig2)/n
lam_x = n*mu2/sig2
lam_x[lam_x<0] = 0

y_ms = (y.mean('trial')-y.mean(('trial', 'stim')))
mu2 = (y_ms**2).sum('stim') - ((m-1)*sig2)/n
lam_y = (n*mu2/sig2)
lam_y[lam_y<0] = 0

snr = (lam_y*lam_x)**0.5
x = x.transpose('unit', 'trial', 'stim').values
y = y.transpose('unit', 'trial', 'stim').values

r2c, r2 = rc.r2c_n2n(x, y)
r2c = r2c.squeeze()
r2_orig = r2.squeeze()
df = pd.read_csv('./figs/fig_data/apc_4_trial_stats.csv')
df['snr'] = snr
df.to_csv('./figs/fig_data/apc_4_trial_stats.csv')

#%%

v4 = xr.open_dataset('/home/dean/Desktop/modules/r2c/data/apc370_with_trials.nc')['resp']
try:
    v4 = v4.rename({'trials':'trial'})
    v4 = v4.rename({'shapes':'stim'})
except:
    print('')
v4.coords['unit'] = range(109)
ys = []
for cell in range(109):
    y = v4.sel(unit=cell).dropna('trial')
    if (y.shape[1])==4:
        ys.append(y)
temp = xr.concat(ys, 'unit')
ys = temp**0.5
m = 371
n = 2
n_sims = int(2e4)
alpha =10
#(ys**2).to_netcdf('./figs/fig_data/apc_4_trial_raw_resp.nc')
nunits = ys.shape[0]


#%%
# now store all of these CI's and seperate out bad fits
# unit X l, h X kind
#unit X r2c_points X l h
#xr.DataArray()
lhs = []
for i in range(ys.shape[0]):
    print(i)
    X = [ys[i,:,1::2].values.T, ys[i, :, ::2].values.T]
    x, y = (ys[i,:,1::2].values.T, ys[i, :, ::2].values.T)
    #est = rc.r2c_b(X)
    ll, ul, r2c_hat_obs, alpha_obs = rc.r2c_ci(x,y, alpha_targ=0.10, nr2cs=100)
#    [l, h, l_np, h_np, l_cdf, h_cdf, 
#                    fixed_r2cs_interp, lh_alphas] = rc.get_lhs(est, m,n, alpha, n_sims)
    lhs.append([ll, ul, alpha_obs] )

#%%
import matplotlib.pyplot as plt

for i, lh in enumerate(lhs):
    X = [ys[i,:,1::2].values.T, ys[i, :, ::2].values.T]
    x, y = (ys[i,:,1::2].values.T, ys[i, :, ::2].values.T)
    est = rc.r2c_b(X)[0]
    plt.figure()
    plt.plot(np.linspace(0,1,50), lh[-1])
    plt.title(lh[:2] + [est,] + [i,])
    
#%%
#for i, lh in enumerate(lhs):
i = 40
X = [ys[i,:,1::2].values.T, ys[i, :, ::2].values.T]
x, y = (ys[i,:,1::2].values.T, ys[i, :, ::2].values.T)
est = rc.r2c_b(X)[0]

r2s = np.linspace(0,1,50)
alpha_targ = 0.1
tol = alpha_targ/4
alpha_obs = lhs[i][-1]
ll_dif = (alpha_obs-alpha_targ/2)<tol# need to fix this
if np.sum(ll_dif)>0:
    ll = r2s[ll_dif][0]
else:
    ll = 0

ul_dif = np.abs(alpha_obs-(1-alpha_targ/2))<tol
if np.sum(ul_dif)>0:
    ul = r2s[ul_dif][-1]
else:
    ul=1
    
#print([ll,ul])
plt.figure()
plt.plot(np.linspace(0,1,50), alpha_obs)
plt.title([ll, ul] + [est,] + [i,])
    
#%%
lam2_bar = ys.mean('trial').var('stim') - ys.var('trial').mean('stim')/len(ys.coords['trial'])
#%%
m = ['r2', 'r2c', 'sig2', 'mux', 'muy', 'lam_bar', 'ci_l', 'ci_h', 'cell_id']
df = pd.DataFrame(np.zeros((nunits, len(m))), columns=m)
for i in range(ys.shape[0]):
    X = [ys[i,:,1::2].values.T, ys[i, :, ::2].values.T]
    l,h = lhs[i][:2]
    est = rc.r2c_b(X)
    df.loc[i][1:5] = est[:4]
    df.loc[i][[-3,-2]] = [l, h]
    df.loc[i][0] = np.corrcoef(X[0].mean(0), X[1].mean(0))[0,1]**2

    
n = len(ys.coords['trial'])/2;m=len(ys.coords['stim'])
df['lam_bar'] = np.sqrt(lam2_bar)
df['z_lam_bar'] = df['lam_bar']/df['sig2']**0.5
fl = n*(m-1)*((df['lam_bar']**2)/df['sig2'])
fl1 = stats.ncf(1, m*(n-1), fl).mean()
c = stats.f(1, m*(n-1)).ppf(0.999)
df['logsf_f'] = stats.f(1, m*(n-1)).logsf(fl1)/np.log(10)
df['cell_id'] = snr.coords['w_lab'].values.astype('str')
#df.to_csv('./figs/fig_data/apc_4_trial_stats_pymc3.csv')
plt.scatter(((c/fl)), df['r2'])
plt.semilogx()
plt.xlim(1e-3,1)
#%%
stats.ncf(1, m*(n-1), fl).logcdf(c)


#%%
from scipy.io import loadmat
ecc = loadmat('./data/yas_inv/ecc_yas_80.mat')['ecc'].squeeze()
dia = ecc*.625 + 40
dia = dia/39
ecc = ecc/39
#%%
ppd = 39
tiscl = []
frfs = []

for i in range(0,80,1):
    da = xr.open_dataarray('data/yas_inv/'+str(i)+'.nc')
    nb = da.stack(c=['rot', 'sc', 'stim_id']).dropna('c', how='all').dropna('trial', how='all')
    
    ti = nb.isel(trial=slice(0,5)).dropna('c', how='any')
    rf = ti.mean('c').mean('trial')
    cid = np.argmax(rf.values)
    rf_inds = np.arange(len(rf)) - cid
    
    deg = ti.coords['pos'].values/ppd
    ti.coords['pos'] = deg - deg[cid]
    #ti.coords['rf_deg'] = ('pos', ti.coords['pos'].values + ecc[i]) 
    #ti.coords['frf'] = ('pos', ti.coords['pos'].values/dia[i]) 
    frfs.append(ti.coords['pos'].values/dia[i])
    ti.coords['pos'] = rf_inds
    tiscl.append(ti)

tiscl2 = xr.concat(tiscl, 'unit') 
tiscl2.coords['unit'] = range(len(tiscl))   
tiscl2 = tiscl2.reset_index('c')
tiscl2.coords['c'] = range(49)
tiscl2.to_netcdf('./figs/fig_data/ti_raw_trial.nc')
ti = tiscl

rti = xr.open_dataarray('./figs/fig_data/ti_raw_trial.nc').load()
#%%
nunits = len(frfs)
omax=0
for rf in frfs:
    pmax = len(rf)
    if pmax>omax:
        omax=pmax
x = np.zeros((omax,nunits))
x[...] = np.nan
for i, rf in enumerate(frfs):
    x[:len(rf),i] = rf
np.save('./figs/fig_data/yas_ti_frf', x)
#%%
measure_names = ['r2', 'r2c', 'sig2', 'mux', 'muy', 'snr', 'ci_l', 'ci_h', 'cell_id']
measure_names = ['r2', 'r2c', 'sig2', 'mux', 'muy', 'snr', 'ci_l', 'ci_h', 'cell_id']

alpha = 10
lhs = []
nunits = len(ti)
#nunits=2
das =[]
for i in np.arange(nunits):
    print(i/nunits)
    t = ti[i].dropna('pos')
    da = xr.DataArray(np.zeros((t.shape[0], len(measure_names))), 
                      dims=['pos', 'm'], coords=[t.coords['pos'], measure_names])
    #da.coords['rf_deg'] = ('pos', t.coords['rf_deg'].values)
    #da.coords['frf'] = ('pos', t.coords['frf'].values)
    for k in range(t.shape[0]):
        x, y = [t.loc[0].values**0.5, t[k].values**0.5]
        m, n = x.shape
        X = [x,y]
        est = rc.r2c_b(X)
        est = np.array(est).squeeze()
        
        ll, ul, r2c_hat_obs, alpha_obs = rc.r2c_ci(x, y, alpha_targ=0.10, nr2cs=100)
        print([ll,ul])
#        [l, h, l_np, h_np, l_cdf, h_cdf, 
#                fixed_r2cs_interp, lh_alphas] = get_lhs(est, m,n, alpha, n_sims)
#        lhs.append([l, h, l_np, h_np, l_cdf, h_cdf, 
#                    fixed_r2cs_interp, lh_alphas] )
        da[k].loc[['r2c', 'sig2', 'mux', 'muy']] = est[:4]
        da[k].loc[['ci_l', 'ci_h']] = [ll, ul]
        da[k].loc['r2'] = np.corrcoef(x.mean(0), y.mean(0))[0,1]**2
        sig2 = est[1]
        mu2x = rect(est[2])
        mu2y = rect(est[3])
        snr = (n*np.sqrt(mu2x*mu2y))/sig2
        da[k].loc['snr'] = snr 
    das.append(da)

#%%  
import r2c_common as rc
x,y = (ti[51][[1,2]]**0.5).values
ll, ul, r2c_hat_obs, alpha_obs = rc.r2c_ci(x, y, alpha_targ=0.10, nr2cs=100)
#%%
plt.plot(alpha_obs)
#%%
das = xr.concat(das, 'unit')
das.coords['unit'] = range(nunits)
das.loc[..., 0,'r2c'] = 1 
#das.to_netcdf('./figs/fig_data/ti_stats_pymc3.nc')
 
#%%
v4 = xr.open_dataset('/home/dean/Desktop/modules/r2c/data/fo.nc')
v4 = v4['__xarray_dataarray_variable__']*.3
print(v4)
nunits = v4.shape[0]
#%%
# now store all of these CI's and seperate out bad fits
# unit X l, h X kind
#unit X r2c_points X l h
#xr.DataArray()
lhs = []
alpha=10
n_sims = int(1e4)

for i in range(nunits):
    print('')
    print(i/nunits)
    a = v4[i].dropna('trial', how='any')**0.5
    #plt.figure()
    #a[0].plot()
    _,m,n = a.shape
    X = [a[0].T.values, a[1].T.values]
    x,y = [a[0].T.values, a[1].T.values]
    ll, ul, r2c_hat_obs, alpha_obs = rc.r2c_ci(x,y, alpha_targ=0.10, nr2cs=100)

    est = rc.r2c_b(X)
    est[2] = rect(est[2])
    est[3] = rect(est[3])
    
    print(np.round(est[:4],2))

    lhs.append([ll, ul, alpha_obs] )
#%%
m = ['r2', 'r2c', 'sig2', 'mux', 'muy', 'snr', 'ci_l', 'ci_h', 'cell_id']
df = pd.DataFrame(np.zeros((nunits, len(m))), columns=m)
for i in range(nunits):
    a = v4[i].dropna('trial', how='any')**0.5
    _,m,n = a.shape
    X = [a[0].values.T, a[1].values.T]
    x,y = [a[0].values.T, a[1].values.T]
    l,h = lhs[i][:2]
    est = rc.r2c_b(X)
    df.loc[i][1:5] = est[:4]
    df.loc[i][[-3,-2]] = [l, h]
    df['r2'][i] = np.corrcoef(x.mean(0), y.mean(0))[0,1]**2
    
    sig2 = est[1]
    mu2x = rect(est[2])
    mu2y = rect(est[3])
    snr = (n*np.sqrt(mu2x*mu2y))/sig2
    df['snr'][i] = snr 

#df.to_csv('./figs/fig_data/fo_stats.csv')


#%%
import os
ophys_dir = '/loc6tb/data/responses/ophys_allen/'
fs = os.listdir(ophys_dir)
fs = [f for f in fs if 'static' in f]

lines = ['Cux2-CreERT2',
 'Emx1-IRES-Cre',
 'Fezf2-CreER',
 'Nr5a1-Cre',
 'Ntsr1-Cre_GN220',
 'Pvalb-IRES-Cre',
 'Rbp4-Cre_KL100',
 'Rorb-IRES2-Cre',
 'Scnn1a-Tg3-Cre',
 'Slc17a7-IRES2-Cre',
 'Sst-IRES-Cre',
 'Tlx3-Cre_PL56',
 'Vip-IRES-Cre']

layers = [[3, 4],
          [3, 4, 5],
          [5,],
          [4,],
          [6,],
          [4, 5],
          [5,],
          [4,],
          [4,],
          [3, 4, 5],
          [4,5],
          [5,],
          [3,4],
         ]

excitatory = [[1,],
          [1,],
          [1,],
          [1,],
          [1,],
          [0,],
          [1,],
          [1,],
          [1,],
          [1,],
          [0,],
          [1,],
          [0,],
         ]
#%% need to find repeats across animals
structures =['VISp_','VISl_', 'VISam_', 'VISpm_', 'VISrl_'] 
import matplotlib.pyplot as plt
depths = [175, 265, 275, 320, 335, 375]
desc = []
for struct in structures[:2]:
    for line in lines:

        for depth in depths:
            nms = [nm for nm in fs if (line in nm) and (struct in nm)
                    and (str(depth) in nm) and 'pos' in nm]

            for nm in nms:
                ds = xr.open_dataset(ophys_dir+nm)
                print(ds.coords['cell'].shape)
                print(nm)
                n_cells = ds.coords['cell'].shape
                desc.append([n_cells, nm])
                
                
#%%
nm = 'staticgrating_Emx1-IRES-Cre_280638_VISp_275'
nm = 'staticgrating_Cux2-CreERT2_222425_VISp_175'   
nm = 'staticgrating_Cux2-CreERT2_229107_VISl_275_id507691735_pos'
nm = 'staticgrating_crelineCux2-CreERT2_donor225037_structVISp_depth175_id502368172_eyeFalse'
snrs = []
dsss = []
ds1s = []
for nm in fs:
    ds1 = xr.open_dataset(ophys_dir+nm) 
    
    dss = ds1['resp'].stack(c=['spatial_frequency', 'orientation', 'phase'])
    dss = dss.dropna('c', how='all')
    dss = dss.dropna('cell', how='all')
    dss = dss.dropna('trial', how='any')
    dss = dss/dss.std('trial')    
    
    n = dss.isnull().mean('trial')
    #plt.imshow(dss[:,0]);plt.colorbar();
    sig2 = 1
    m = len(dss.coords['c'])
    n = len(dss.coords['trial'])
    mu2 = (dss.mean('trial')**2).sum('c') - ((m-1)*sig2)/n
    snr = (mu2/m)**0.5
    snrs.append((np.round(np.nanmean(snr.values>0.37),2)))
    ds1s.append(ds1)
    dsss.append(dss)
    print(np.nansum(mod_p(m,n,snr)<0.05))
    #%%
best_snr = np.argmax(snrs)
dss = dsss[best_snr]
ds1 = ds1s[best_snr]

#%%
n_trial = len(dss.coords['trial'])
m_stim = len(dss.coords['c'])

m = dss - dss.mean('trial')
m = m/(m**2).sum('trial')**0.5
m = m.transpose('cell', 'c', 'trial').values
m = m.reshape((m.shape[0], np.product(m.shape[1:])))
noise_cor = np.dot(m, m.T)/m_stim

m = dss.mean('trial')

signal_cor = np.corrcoef(m)


mu2 = (m**2).sum('c') - ((m_stim-1)*sig2)/n_trial
snr = (mu2/m_stim)**0.5
#%%
p_signal_cor = signal_cor.copy()
p_signal_cor[np.diag_indices_from(p_signal_cor)] = np.nan
plt.scatter(np.triu(p_signal_cor), np.triu(noise_cor))
stats.pearsonr(np.triu(p_signal_cor, 1).ravel(), 
               np.triu(noise_cor, 1).ravel())

#%% now simulate

a1 = 1
a2 = 2

n1 = 2
n2 = 2

d1 = 2
d2 = 2

r_n = 0.9**0.5
r_s = 0.1**0.5

t = 1

n = 10
m = 100
n_sims = 5000

S = r_s*np.random.normal(loc=0, scale=1, size=(1, m, n_sims)) 

S1 = d1*S
R1 = d1*((1-r_s**2)**0.5)*np.random.normal(loc=0, scale=1, size=(1, m, n_sims)) 

S2 = d2*S
R2 = d2*((1-r_s**2)**0.5)*np.random.normal(loc=0, scale=1, size=(1, m, n_sims))

C =  (r_n)*np.random.normal(loc=0, scale=1, size=(n, m, n_sims))
C1 =  n1*C
N1 =  n1*((1-r_n**2)**0.5)*np.random.normal(loc=0, scale=1, size=(n, m, n_sims))

C2 =  n2*C
N2 =  n2*((1-r_n**2)**0.5)*np.random.normal(loc=0, scale=1, size=(n, m, n_sims))


R_1 = np.sqrt(t)*(a1 + S1 + R1 + C1) + N1
R_2 = np.sqrt(t)*(a2 + S2 + R2 + C2) + N2

pred_sig_rho_raw = (d1*d2*r_s**2 + n1*n2*r_n**2)/((d1**2+n1**2)*(d2**2+n2**2))**0.5
sig_cov_raw = np.array([np.cov(R_1[..., i].ravel(), R_2[..., i].ravel()) for i in range(n_sims)])
print([(sig_cov_raw[:,0,1]/(sig_cov_raw[:,0,0]*sig_cov_raw[:,1,1])**0.5).mean(), 
       pred_sig_rho_raw])

pred_sig_rho_avg = (d1*d2*r_s**2 + n1*n2*r_n**2/n)/((d1**2+n1**2/n)*(d2**2+n2**2/n))**0.5
sig_cov_avg = np.array([np.cov(R_1[..., i].mean(0), R_2[..., i].mean(0)) for i in range(n_sims)])
print([(sig_cov_avg[:,0,1]/(sig_cov_avg[:,0,0]*sig_cov_avg[:,1,1])**0.5).mean(), pred_sig_rho_avg]) 


pred_sig_rho_avg = (d1*d2*r_s**2 + n1*n2*r_n**2/n)/((d1**2+n1**2/n)*(d2**2+n2**2/n))**0.5

#%%


dists = ds1['cell_pos'].dropna('cell')

dists.shape
dist_difs = dists.values[...,np.newaxis] - dists.values[:,np.newaxis]
dist_mat = (dist_difs**2).sum(0)**0.5
#%%

t = 1
ds_ms = dss - dss.mean('trial')
a = ds_ms.transpose('cell', 'trial', 'c').values
a = a[:]
b = dss.transpose('cell', 'trial', 'c').values
nrs = np.zeros((a.shape[0], a.shape[0]))
srs = np.zeros((a.shape[0], a.shape[0]))            
srs_o = np.zeros((a.shape[0], a.shape[0]))
srs_osh = np.zeros((a.shape[0], a.shape[0]))
for i in range(a.shape[0]):
    for j in range(a.shape[0]):
        if i>=j:
            c1 = a[i]
            c2 = a[j]
            
            cov_sc_hat = np.array([np.cov(c1[:,k], c2[:,k]) 
                                    for k in range(c1.shape[-1])]).mean(0)
            
            cov_sc_hat_norm = cov_sc_hat[0,1]/t
            
            sig2_n_hat = 1 - cov_sc_hat[0,1]
            r_sc_hat_norm  = (cov_sc_hat_norm)/(cov_sc_hat_norm + sig2_n_hat)
            nrs[i,j] = r_sc_hat_norm
            
            c1 = b[i]
            c2 = b[j]
            mid = int(c1.shape[0]/2.)
            srs[i,j] =( rc.r2c_n2n(c1[::2],c2[1::2])[0].squeeze()+
                       rc.r2c_n2n(c1[1::2],c2[::2])[0].squeeze())/2
            srs_o[i,j] = (rc.r2c_n2n(c1,c2)[1].squeeze())
            srs_osh[i,j] = ( rc.r2c_n2n(c1[::2],c2[1::2])[1].squeeze()+
                       rc.r2c_n2n(c1[1::2],c2[::2])[1].squeeze())/2


#%%

thresh=0.2
split_half_r2c = np.diag(srs)
split_half_r2o = np.diag(srs_osh)
inds = split_half_r2o>thresh
n1 = np.arange(len(inds))[inds]
srs_temp = srs.copy()[np.ix_(n1, n1)]
srs_osh_temp = srs_osh.copy()[np.ix_(n1, n1)]
srs_o_temp = srs_o.copy()[np.ix_(n1, n1)]
nrs_temp = nrs.copy()[np.ix_(n1, n1)]
#%%

#%%   
nunits = srs_temp.shape[0]
dist_mat_sub = dist_mat[:nunits,:nunits]
  
srs_o_temp[np.diag_indices_from(srs_o_temp)] = 0
srs_temp[np.diag_indices_from(srs_temp)] = 0
nrs_temp[np.diag_indices_from(nrs_temp)] = 0


dists_rav = dist_mat_sub[~(srs_temp==0)]
srs_rav = srs_temp[~(srs_temp==0)]

nrs_rav = nrs_temp[~(nrs_temp==0)]

srs_o_rav = srs_o_temp[~(srs_o_temp==0)]
#%%
s=3
plt.figure(figsize=(9,3))
plt.subplot(131)
r,p = np.round(stats.spearmanr(split_half_r2o[inds], 
                               split_half_r2c[inds]),2)
plt.title(r'$r_s=$' +str(r) + ', '+ r'$p=$' +str(p))
plt.scatter(split_half_r2o, split_half_r2c, s=s)
plt.ylim([0,2]);plt.xlim(-.2,1)
plt.ylabel('Split-half $\hat{R}^2_c$')
plt.xlabel('Split-half $\hat{R}^2$')
plt.plot([thresh,thresh], [0,2])
plt.annotate('Noise threshold', (thresh*1.04,0.5))
plt.plot([-.2,1],[1,1], c='k', alpha=0.5)

plt.subplot(132)
plt.scatter(nrs_rav, srs_rav, s=s);
plt.ylim([0,1.5]);plt.xlim(0,.5)
plt.ylabel('Split-half $\hat{R}^2_c$')
plt.xlabel('$\hat{R}_{SC}$')

#plt.scatter(nrs, srs_o);
r,p = np.round(stats.spearmanr(nrs_rav, srs_rav),2)
plt.title(r'$r_s=$' +str(r) + ', '+ r'$p=$' +str(p))
(r'$r_s=$' +str(r))

plt.subplot(133)
r,p = np.round(stats.spearmanr(dists_rav, srs_rav),2)
plt.title(r'$r_s=$' +str(r) + ', '+ r'$p=$' +str(p))
plt.scatter(dists_rav, srs_rav,s=s);plt.ylim(0,1.5)
plt.ylabel('Split-half $\hat{R}^2_c$')

plt.xlabel('Distance between cell pairs '+ r'($\mu m$)')
plt.tight_layout()
plt.savefig('/home/dean/Desktop/modules/r2c/figs/allen_signal_noise_corr.pdf')
#%%
print(stats.spearmanr(nrs_rav, srs_o_rav))
plt.scatter(nrs_rav, srs_o_rav)
#%%
print(stats.spearmanr(dists_rav, srs_o_rav))
plt.scatter(dists_rav, srs_o_rav)

#%%
from scipy import optimize
from scipy import stats
def mod_p(m,n,snr):
    c = stats.f(n*m-1, m*(n-1)).ppf(0.99)
    p = stats.ncf(n*m-1, m*(n-1), nc=(n*m)*(snr)).cdf(c)
    return p

alpha = 0.01

def make_mod_p(snr):
    return lambda x: np.abs(stats.ncf(x[1]*x[0]-1, x[0]*(x[1]-1), 
                                      nc=(x[1]*x[0])*(snr)).cdf(
                                              stats.f(x[1]*x[0]-1, 
                                              x[0]*(x[1]-1)).ppf(0.99)) - alpha)

def make_mod_pn(snr,n):
    return lambda x: np.abs(stats.ncf(n*x[0]-1, x[0]*(n-1), 
                                      nc=(n*x[0])*(snr)).cdf(
                                              stats.f(n*x[0]-1, 
                                              x[0]*(n-1)).ppf(0.99)) - alpha)


snrs = np.logspace(-1,1,50)
#mns = []
#m=4
#n=4
#for asnr in snrs:
#    a = make_mod_p(asnr)
#    r = optimize.minimize(a, (m,n), tol=0.01, method='nelder-mead')
#    mns.append((r['x'][0], r['x'][1]))
#    #print(mod_p(r['x'][0], r['x'][1], asnr))

ms = []
m=1
n=4
for asnr in snrs:
    a = make_mod_pn(asnr,n)
    r = optimize.minimize(a, (m,), tol=0.01, method='nelder-mead')
    ms.append((r['x'][0]))
    #print(mod_p(r['x'][0], r['x'][1], asnr))

ms = np.array(ms)*n 


mns = []
m=1
n=40
for asnr in snrs:
    a = make_mod_pn(asnr,n)
    r = optimize.minimize(a, (m,), tol=0.01, method='nelder-mead')
    mns.append((r['x'][0]))
    #print(mod_p(r['x'][0], r['x'][1], asnr))
mns = np.array(mns)*n
#%%
plt.title('Power analysis F-test effect of stimuli')
plt.plot(snrs, mns, c='k')
plt.plot(snrs, ms, c='r')
plt.legend(['n=40', 'n=4'], loc='lower left', title='Repeats')
plt.semilogy()
plt.xlabel('SNR')
plt.ylabel('Total trials for ' + r'$P(F_{nm-1,m(n-1)}(mn{(SNR)^2)}>c(0.99)| H_1)=0.99$')
plt.ylabel('Total trials for ' + r'$P(F>F_c(0.99)| H_1)=0.99$')

#plt.ylim(1,100000)
#plt.xlim(1,10)
plt.yticks([1,10,100,1000,10000, 100000])
plt.xticks([0,0.25,0.5,0.75,1,1.25, 1.5])
plt.grid()
plt.plot([snrs[0], snrs[-1]],[1200, 1200])
plt.plot([snrs[0], snrs[-1]],[5500, 5500])

plt.annotate('Total trials Allen NP grating data ~5500', (.5,5500+500))
plt.annotate('Total trials MT dot motion data ~1200', (.5,1200+100))
plt.semilogx()

plt.savefig('./figs/power_an_snr_f.pdf')

print(snrs[np.argmin(np.abs(mns - 5500))])
print(snrs[np.argmin(np.abs(mns - 1200))])


#%%
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
top_dir = '/loc6tb/data/responses/ophys_allen/boc/'
boc = BrainObservatoryCache(cache=True, 
                            manifest_file='/loc6tb/data/responses/ophys_allen/boc/brain_observatory_manifest.json')
lines = boc.get_all_cre_lines()
depths = boc.get_all_imaging_depths()
structs = boc.get_all_targeted_structures()
stim = boc.get_all_stimuli()

line = lines[0]
depth = depths[0]
struct = structs[0]

exps = boc.get_ophys_experiments(
                  targeted_structures=['VISp',],
                  stimuli=['static_gratings'],
                  require_eye_tracking=False)

ids = [exp['id']for exp in exps]

load_dir = '/loc6tb/data/responses/ophys_allen/static_stim_resp/grating/'
visp_ca = [xr.open_dataset(load_dir + str(int(an_id)) + '.nc')['__xarray_dataarray_variable__']
 for an_id in ids]

#%%
#load np allen 
load_dir = '/loc6tb/data/responses/np_allen/grating/'
fns = os.listdir(load_dir)
fns = [fn for fn in fns if '.nc' in fn ]
visp_np = [xr.open_dataset(load_dir+fn).load() for  fn in fns]
visp_np = [visp_ for visp_ in visp_np if len(visp_)>0]
n_recs = len(visp_np)
#visp = xr.concat(visp, 'rec')

for i, visp_ in enumerate(visp_np):
    
    visp_ = visp_['spike_counts']
    area = visp_.coords['area']
    visp_ = visp_[area=='VISp']
    visp_.coords['unit_id'] = range(len(visp_.coords['unit_id']))
    visp_np[i] = visp_
    

visp_np = xr.concat(visp_np, 'rec')
visp_np.coords['rec'] = range(n_recs)

#%%
# load wyeth mt

load_dir = '/loc6tb/data/responses/v1dotd/'
fns = os.listdir(load_dir)
fns = [fn for fn in fns if '.nc' in fn ]

mt = [xr.open_dataarray(load_dir+fn).load() for  fn in fns]
fns = [fn.split('.')[0] for fn in fns]
nms = [amt.name for amt in mt]
mt = xr.concat(mt, 'rec')
mt.coords['rec'] = range(len(fns))

mt.coords['nms'] = ('rec', nms)
mts = mt.sel(t=slice(0, 2), unit=slice(0,1))

s = mts.sum('t', min_count=1)**0.5
s = s[~(s.coords['nms']=='emu089')]
#%% now we fill up cor data
w = pd.read_csv('/home/dean/Desktop/modules/r2c/data/wyeth_r_cc_mt_vals.dat', 
            header=None, delim_whitespace=True, index_col=0)
w.columns = ['r_s', 'r_n']

w = w.drop([ 'emu019.02',
 'emu019.12', ])
w = w.rename(index={'emu037.01':'emu037', 'emu019.01':'emu019'})
#%% go through s and get relevant data 
ms = ['w_r_n', 'll', 'ul', 'r2c', 'r2', 'snr']
cor_dat = pd.DataFrame(np.zeros((len(s), len(ms))), columns=ms, 
                       index= s.coords['nms'].values)
#%%

ll, ul, r2c_hat_obs, alpha_obs = [np.nan,]*4
for rec in range(len(s)):
    print(rec/len(s))
    s_temp = s[rec].dropna('trial_tot')
    u1, u2 = s_temp
    n, m = u1.shape
    mid = int(np.floor(n/2))
    ll, ul, r2c_hat_obs, alpha_obs = rc.r2c_ci(u1.values[:mid], 
                                           u2.values[mid:], 
                                           alpha_targ=0.10, nr2cs=100)
    r2 = np.corrcoef(u1[:mid].mean('trial_tot').values, 
                u2[mid:].mean('trial_tot').values)[0,1]**2
                     
    snr = np.double((s_temp.mean('trial_tot').var('dir')/
           s_temp.var('trial_tot').mean('dir')).prod('unit').values)
    cor_dat.iloc[rec] = np.array([w.loc[s[rec].coords['nms'].values]['r_n'], 
                                 ll, ul, r2c_hat_obs, r2, snr]).squeeze()


#%%
s.coords['rec'] = s.coords['nms']
hi_r2_lo_snr = (cor_dat['r2'].rank()-
                cor_dat['snr'].rank()).sort_values().index.values[::-1]

hi_r2_hi_snr = (cor_dat['r2'].rank()+
                cor_dat['snr'].rank()).sort_values().index.values[::-1]

lo_r2_lo_snr = (-cor_dat['r2'].rank()-
                cor_dat['snr'].rank()).sort_values().index.values[::-1]

lo_r2_hi_snr = (-cor_dat['r2'].rank()+
                cor_dat['snr'].rank()).sort_values().index.values[::-1]


#%%
import matplotlib.pyplot as plt

i=0
plt.figure(figsize=(8,8))
for ranking in [hi_r2_hi_snr, lo_r2_hi_snr,  hi_r2_lo_snr, lo_r2_lo_snr]:
    for nm in ranking[:3]:
       i+=1
       plt.subplot(4, 3, i)
        
       u1, u2 = (s.loc[nm].dropna('trial_tot')**2)/2
       n,m = u1.shape
       plt.errorbar(x=u1.coords['dir'].values, 
                    y=u1.mean('trial_tot'),
                    yerr=u1.std('trial_tot')/(n**0.5))
       
       plt.errorbar(x=u1.coords['dir'].values, 
                    y=u2.mean('trial_tot'),
                    yerr=u2.std('trial_tot')/(n**0.5))
       #plt.ylim(0,110)

       if i==10:
           plt.xlabel('Motion direction (deg)')
           plt.ylabel('spk/s')
       else:
           plt.gca().set_xticklabels([])
           plt.gca().set_yticklabels([])
       r_n, ll, ul, r2c_hat_obs, r2, snr = cor_dat.loc[nm].values
       plt.title(nm+'\n$r^2_{ER}=$' + str(np.round(r2c_hat_obs, 2)) +
                  ', CI $\in$'+ str([np.round(ll),np.round(ul)]) + 
                 ',\n$r^2=$' + str(np.round(r2, 2)) + 
                 ', SNR=' + str(np.round(snr,2)), fontsize=8)
#       plt.legend(['unit 1', 'unit 2'])
plt.tight_layout()

#%%

import pandas as pd
import xarray as xr

cor_dat = pd.read_csv("/Users/deanpospisil/Desktop/modules/r2c/data/r2c_ci_cor_dat.cvs",
                      index_col=0)
cor_dat['cilen']  = 1-(cor_dat['ul'] - cor_dat['ll'])
s = xr.open_dataarray('/Users/deanpospisil/Desktop/modules/r2c/data/mt_dot_sum_sqrt.nc' )
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
plt.savefig('/Users/deanpospisil/Desktop/modules/r2c/figs/r2c_vs_r2_cithresh.pdf');
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
plt.savefig('/Users/deanpospisil/Desktop/modules/r2c/figs/r2c_vs_r2_cithresh_examples.pdf');

#%%
from scipy import  stats
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
plt.savefig('/Users/deanpospisil/Desktop/modules/r2c/figs/r2c_vs_snr_cithresh.pdf');

#%%
nm = 'emu047'
u1, u2 = (s.loc[nm].dropna('trial_tot')**2)/2
n,m = u1.shape
plt.errorbar(x=u1.coords['dir'].values, 
y=u1.mean('trial_tot'),
yerr=u1.std('trial_tot')/(n**0.5))
   
plt.errorbar(x=u1.coords['dir'].values, 
y=u2.mean('trial_tot'),
yerr=u2.std('trial_tot')/(n**0.5))


