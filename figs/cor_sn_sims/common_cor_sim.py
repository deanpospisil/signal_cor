#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:14:17 2020

@author: dean
"""
import numpy as np


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
def n2n_pearson_r_sin(r, m):

    angle = np.arccos(r)  
    s = np.linspace(0, 2*np.pi, int(m))
   
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
    p_cell_pairs = 1
    if r_n==0:
        r_n = r_n
    else:
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
    X = X_s[np.newaxis, :, np.newaxis, np.newaxis] + X_n
    
    Y_n =    n_y*((abs(r_n)**0.5)*C + ((1-abs(r_n))**0.5)*N_y)
    Y = Y_s[np.newaxis, :, np.newaxis, np.newaxis] + Y_n
    
    hat_r_n = noise_cor_fast(X, Y)
    
    if split:
        hat_r_s = sig_cor_fast_split(X, Y)
    else:    
        hat_r_s = sig_cor_fast(X, Y)
    
    
    return Y_s, X_s, Y, X, hat_r_n, hat_r_s


def gen_sig_noise_cor_rand_sim(rho_rs_rn, n_sims, p_cell_pairs, sig_n, sig_s, 
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

def gen_a_sig_noise_cor_rand_sim( n_sims, 
                          snr, n, m, r_n, r_s,
                          n_x, n_y, split=False):

    a_x = a_y = 1
    p_cell_pairs = 1

    
    #fixed
    if not r_n==0:
        n_y = np.sign(r_n)*n_y
    
    #derived
    d_x = (snr*np.abs(n_x*n_y))**0.5
    d_y = d_x.copy()
    
    if not r_s==0:
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

    
    return hat_r_s, hat_r_n



def hat_r_s_naive_rand_stim(r_n, r_s, d_x, d_y, n_x, n_y, n):
    hat_r_s = ((r_s*(d_x*d_y) + (r_n*(n_x*n_y))/n)/(d_x*d_y + (n_x*n_y)/n))
    return hat_r_s


def hat_r_s_naive_fixed_stim(r_n, r_s, d_x, d_y, n_x, n_y, n, m):
    num_pred = (r_n*m*(n_x*n_y)/n + d_x*d_y*r_s)
    den_pred = (4*r_n*(n_x*n_y/n)*d_x*d_y*r_s + 2*m*(r_n*(n_x*n_y/n))**2
       + ((d_x**2) + m*(n_x**2)/n)*(d_y**2 + m*(n_y**2)/n))**0.5
    return num_pred/den_pred
