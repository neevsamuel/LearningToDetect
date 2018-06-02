#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import time as tm
import math
import sys
import pickle as pkl
from copy import deepcopy


def find_nearest_mpsk_np(values):
    min_dist = 999
    ret_Val = [0, 0]
    dist = np.square(values[0] - 1)+ np.square(values[1] - 0)
    if dist < min_dist:
        min_dist=dist
        ret_Val = [1,0]
    dist = np.square(values[0] - 0)+ np.square(values[1] - 1)
    if dist < min_dist:
        min_dist=dist
        ret_Val = [0,1]
    dist = np.square(values[0] - (-1))+ np.square(values[1] - 0)
    if dist < min_dist:
        min_dist=dist
        ret_Val = [-1,0]
    dist = np.square(values[0] - 0)+ np.square(values[1] - (-1))
    if dist < min_dist:
        min_dist=dist
        ret_Val = [0,-1]

    dist = np.square(values[0] - 1.0/np.sqrt(2))+ np.square(values[1] - 1.0/np.sqrt(2))
    if dist < min_dist:
        min_dist=dist
        ret_Val = [1.0/np.sqrt(2),1.0/np.sqrt(2)]

    dist = np.square(values[0] + 1.0/np.sqrt(2))+ np.square(values[1] - 1.0/np.sqrt(2))
    if dist < min_dist:
        min_dist=dist
        ret_Val = [-1.0/np.sqrt(2),1.0/np.sqrt(2)]

    dist = np.square(values[0] - 1.0/np.sqrt(2))+ np.square(values[1] + 1.0/np.sqrt(2))
    if dist < min_dist:
        min_dist=dist
        ret_Val = [1.0/np.sqrt(2),-1.0/np.sqrt(2)]

    dist = np.square(values[0] + 1.0/np.sqrt(2))+ np.square(values[1] + 1.0/np.sqrt(2))
    if dist < min_dist:
        ret_Val = [-1.0/np.sqrt(2),-1.0/np.sqrt(2)]

    return ret_Val

def sdr_ip2(z_complex, G_complex,real_x):
    K=np.shape(G_complex)[1]
    G_complex = np.matrix(G_complex)
    z_complex = np.matrix(z_complex)
    H_complex = np.dot(G_complex.getH(),G_complex)
    y_complex = np.dot(G_complex.getH(),z_complex.T)
    epsilon = 1e-5
    Q_complex = np.array(np.bmat([[np.dot(G_complex.getH(),G_complex),  -1*np.dot(G_complex.getH(), z_complex.T)], [-1*np.dot(z_complex.getH().T,G_complex), np.matrix(0)]]))
    Q_complex = -Q_complex

    Q_r = Q_complex.real
    Q_i = Q_complex.imag

    Q = np.concatenate((np.concatenate((Q_r, -1 * Q_i), axis=1),np.concatenate((Q_i, Q_r), axis=1)), axis=0)
    n = np.shape(Q_complex)[1]

    X = 0.5*np.identity(2*n)

    lambd = 1.1*(np.dot(np.absolute(Q_r),np.ones((n,1)))+np.dot(np.absolute(Q_i),np.ones((n,1))))
    Z = np.diag(np.reshape(np.concatenate((lambd,lambd),axis=0),2*n)) - Q
    k = 0
    while(np.trace(np.dot(Z,X)) > epsilon and k<100):

        k=k+1
        mu = np.trace(np.dot(Z,X))/(2*n) #8psk add n->2n
        mu = mu/2
        #compute newton search direction
        W = np.linalg.inv(Z)
        T = np.multiply(W,X)
        T2 = T[0:n,0:n] + T[0:n,n:2*n] + T[n:2*n,0:n]+ T[n:2*n,n:2*n]
        W2 = np.diag(W)[0:n] + np.diag(W)[n:2*n]
        dlambd = np.dot(np.linalg.inv(T2),np.reshape(mu*W2- np.ones(n) ,(n,1)))
        dlambd2 = np.concatenate((dlambd,dlambd),axis=0)
        dZ = np.diag(np.reshape(dlambd2,2*n,1))
        dX = mu*W - X - np.dot(W,np.dot(dZ,X))
        dX = 0.5*(dX + dX.T)

        #line search
        ap = 0.9
        ad = 0.9
        tau = 0.99
        j=1
        try:
            R = np.linalg.cholesky(X + ap*dX) # should be matrix!!!!
            s = 0
        except np.linalg.linalg.LinAlgError:
            s = 1
        while(s==1):
            j += 1
            ap = tau * ( ap ** j )
            try:
                R = np.linalg.cholesky(X + ap*dX)
                s = 0
            except np.linalg.linalg.LinAlgError:
                s = 1
        X = X + ap*dX
        j=1;
        try:
            R = np.linalg.cholesky(Z + ad*dZ)
            s = 0
        except np.linalg.linalg.LinAlgError:
            s = 1
        while(s==1):
            j += 1
            ad = tau * ( ad ** j )
            try:
                R = np.linalg.cholesky(Z + ad*dZ)
                s = 0
            except np.linalg.linalg.LinAlgError:
                s = 1
        Z = Z + ad*dZ;
        lambd = lambd + ad*dlambd


    X_final = X[0:n,0:n] + X[n:2*n,n:2*n]  +1j*X[n:2*n,0:n] -1j*X[0:n,n:2*n]

    x = (X_final[range(K),-1]).T

    return(x)


def batch_sdr2(y,H,x):
    B = np.shape(y)[0]
    ber=0.0
    ser = 0.0
    for i in range(B):
        y_complex = y[i][0:N]   + 1j*y[i][N:2*N]
        H_complex = H[i][0:N,0:K] +  1j*H[i][N:2*N,0:K]
        xx = sdr_ip2(y_complex,H_complex,x[i])
        for ii in range(K):
            x_sdr = xx[ii]
            x_sdr2 = find_nearest_mpsk_np([np.real(x_sdr),np.imag(x_sdr)])
            x_true = [x[i][ii] ,x[i][ii+K]]
            temp_ber,temp_ser = compare_complex(x_sdr2,x_true)
            ber = ber+temp_ber
            ser = ser+temp_ser
    ber = ber/(B*2)
    ser = ser/B
    return np.float32(ber),np.float32(ser)


def compare_complex(x_sdr,x_true):
    real = np.greater(np.abs(x_sdr[0] - x_true[0]),0.1)
    imag = np.greater(np.abs(x_sdr[1] - x_true[1]),0.1)
    ber = real.astype(np.float32)+imag.astype(np.float32)
    ser = (np.greater(ber,0.1)).astype(np.float32)

    return ber,ser

def are_not_equal_mpsk_tf(vals_true, vals_pred):
    return tf.less(0.1,tf.abs(vals_true-vals_pred))

def are_not_equal_mpsk_np(vals_true, vals_pred):
    return np.less(0.1,np.abs(vals_true-vals_pred))



def dfe(y,H):
    Q,R = np.linalg.qr(H)
    Qy = np.dot (Q.T,y)
    xx=np.zeros([2*K])
    for k in range(2*K-1,-1,-1):
        xx[k]=find_nearest_mpsk_np((Qy[k]-np.dot(R[k][k:],xx[k:]))/R[k][k],1,K)
    return(xx)

def batch_dfe(y,H,x,x_R,x_I):
    B = np.shape(y)[0]
    ber=0
    for i in range(B):
        xx = dfe(y[i].T,H[i])
        retValReal = xx[0:K]
        retValIm = xx[K:2 * K]
        ber+=(np.mean(np.logical_or(are_not_equal_mpsk_np(x_R[i], retValReal), are_not_equal_mpsk_np(x_I[i], retValIm))))
    ber=ber/B
    return np.float32(ber)


###start here
sess = tf.InteractiveSession()
#parameters
K = 15
N = 25
snrdb_low = 18.0
snrdb_high = 25.0
snr_low = 10.0 ** (snrdb_low/10.0)
snr_high = 10.0 ** (snrdb_high/10.0)
L=30
v_size = 4*(2*K)
hl_size = 12*(2*K)
startingLearningRate1 = 0.0005
startingLearningRate2 = 0.0005
decay_factor1 = 0.97
decay_factor2 = 0.97
decay_step_size1 = 1000
decay_step_size2 = 1000
train_iter = 2
train_iter_no_noise = 1
n0 = 0.5

train_batch_size = 2000
test_iter= 2
test_batch_size = 2
LOG_LOSS = 1
res_alpha=0.9
num_snr = 6
snrdb_low_test=19.0
snrdb_high_test=24.0

print(K)
print(N)
print(snrdb_low)
print(snrdb_high)
print(snr_low)
print(snr_high)
print(L)
print(v_size)
print(hl_size)
print(startingLearningRate1)
print(startingLearningRate2)
print(decay_factor1)
print(decay_factor2)
print(decay_step_size1)
print(decay_step_size2)
print(train_iter)
print(train_iter_no_noise)
print(n0)
print(train_batch_size)
print(test_iter)
print(test_batch_size)
print(LOG_LOSS)
print(res_alpha)
print(num_snr)
print(snrdb_low_test)
print(snrdb_high_test)

"""Data generation for train and test phases
In this example, both functions are the same.
This duplication is in order to easily allow testing cases where the test is over different distributions of data than in the training phase.
e.g. training over gaussian i.i.d. channels and testing over a specific constant channel.
currently both test and train are over i.i.d gaussian channel.
"""
def generate_data_iid_test_no_noise(B,K,N,snr_low,snr_high):
    x_bit = np.random.randint(2, size=(B, 3 * K))
    x_r = np.zeros((B, K))
    x_i = np.zeros((B, K))
    x_ind = np.zeros((B, 8*K))
    for i in range(B):
        for ii in range(K):
            sym = x_bit[i, 3 * ii:3 * ii + 3]
            if sym[0] == 0 and sym[1] == 0 and sym[2] == 0:
                x_r[i, ii] = -1.0/np.sqrt(2)
                x_i[i, ii] = -1.0/np.sqrt(2)
                x_ind[i,0+ii*8] = 1
            if sym[0] == 0 and sym[1] == 0 and sym[2] == 1:
                x_r[i, ii] = -1
                x_i[i, ii] = 0
                x_ind[i,1+ii*8] = 1
            if sym[0] == 0 and sym[1] == 1 and sym[2] == 1:
                x_r[i, ii] = -1.0/np.sqrt(2)
                x_i[i, ii] = 1.0/np.sqrt(2)
                x_ind[i,2+ii*8] = 1
            if sym[0] == 0 and sym[1] == 1 and sym[2] == 0:
                x_r[i, ii] = 0
                x_i[i, ii] = 1
                x_ind[i,3+ii*8] = 1
            if sym[0] == 1 and sym[1] == 1 and sym[2] == 0:
                x_r[i, ii] = 1.0/np.sqrt(2)
                x_i[i, ii] = 1.0/np.sqrt(2)
                x_ind[i,4+ii*8] = 1
            if sym[0] == 1 and sym[1] == 1 and sym[2] == 1:
                x_r[i, ii] = 1
                x_i[i, ii] = 0
                x_ind[i,5+ii*8] = 1
            if sym[0] == 1 and sym[1] == 0 and sym[2] == 1:
                x_r[i, ii] = 1.0/np.sqrt(2)
                x_i[i, ii] = -1.0/np.sqrt(2)
                x_ind[i,6+ii*8] = 1
            if sym[0] == 1 and sym[1] == 0 and sym[2] == 0:
                x_r[i, ii] = 0
                x_i[i, ii] = -1
                x_ind[i,7+ii*8] = 1


    x_ = np.concatenate((x_r, x_i), axis=1)


    H_R = np.random.randn(B,N,K)
    H_I = np.random.randn(B,N,K)
    H_  = np.zeros([B,2*N,2*K])

    y_=np.zeros([B,2*N])

    w_R = np.random.randn(B,N)
    w_I = np.random.randn(B,N)
    w   = np.concatenate((w_R , w_I) , axis = 1)

    Hy_=x_*0
    HH_=np.zeros([B,2*K,2*K])
    SNR_= np.zeros([B])
    for i in range(B):
        #print i
        SNR = np.random.uniform(low=snr_low,high=snr_high)
        H = np.concatenate((np.concatenate((H_R[i, :, :], -1 * H_I[i, :, :]), axis=1),np.concatenate((H_I[i, :, :], H_R[i, :, :]), axis=1)), axis=0)
        tmp_snr=(H.T.dot(H)).trace()/(2*K)
        H=H/np.sqrt(tmp_snr)*np.sqrt(SNR)
        H_[i,:,:]=H
        y_[i,:]=H.dot(x_[i,:]) #+w[i,:]
        Hy_[i,:]=H.T.dot(y_[i,:])
        HH_[i,:,:]=H.T.dot( H_[i,:,:])
        SNR_[i] = SNR
    return y_,H_,Hy_,HH_,x_,SNR_, H_R, H_I, x_r, x_i, w_R, w_I,x_ind

def generate_data_train_no_noise(B,K,N,snr_low,snr_high):

    x_bit = np.random.randint(2, size=(B, 3*K))
    x_r = np.zeros((B,K))
    x_i = np.zeros((B,K))
    x_ind = np.zeros((B, 8*K))
    for i in range(B):
        for ii in range(K):
            sym = x_bit[i, 3 * ii:3 * ii + 3]
            if sym[0] == 0 and sym[1] == 0 and sym[2] == 0:
                x_r[i, ii] = -1.0/np.sqrt(2)
                x_i[i, ii] = -1.0/np.sqrt(2)
                x_ind[i,0+ii*8] = 1
            if sym[0] == 0 and sym[1] == 0 and sym[2] == 1:
                x_r[i, ii] = -1
                x_i[i, ii] = 0
                x_ind[i,1+ii*8] = 1
            if sym[0] == 0 and sym[1] == 1 and sym[2] == 1:
                x_r[i, ii] = -1.0/np.sqrt(2)
                x_i[i, ii] = 1.0/np.sqrt(2)
                x_ind[i,2+ii*8] = 1
            if sym[0] == 0 and sym[1] == 1 and sym[2] == 0:
                x_r[i, ii] = 0
                x_i[i, ii] = 1
                x_ind[i,3+ii*8] = 1
            if sym[0] == 1 and sym[1] == 1 and sym[2] == 0:
                x_r[i, ii] = 1.0/np.sqrt(2)
                x_i[i, ii] = 1.0/np.sqrt(2)
                x_ind[i,4+ii*8] = 1
            if sym[0] == 1 and sym[1] == 1 and sym[2] == 1:
                x_r[i, ii] = 1
                x_i[i, ii] = 0
                x_ind[i,5+ii*8] = 1
            if sym[0] == 1 and sym[1] == 0 and sym[2] == 1:
                x_r[i, ii] = 1.0/np.sqrt(2)
                x_i[i, ii] = -1.0/np.sqrt(2)
                x_ind[i,6+ii*8] = 1
            if sym[0] == 1 and sym[1] == 0 and sym[2] == 0:
                x_r[i, ii] = 0
                x_i[i, ii] = -1
                x_ind[i,7+ii*8] = 1
    x_ = np.concatenate((x_r, x_i), axis=1)

    H_R = np.random.randn(B, N, K)
    H_I = np.random.randn(B, N, K)
    H_ = np.zeros([B, 2 * N, 2 * K])

    y_ = np.zeros([B, 2 * N])

    w_R = np.random.randn(B, N)
    w_I = np.random.randn(B, N)
    w = np.concatenate((w_R, w_I), axis=1)

    Hy_ = x_ * 0
    HH_ = np.zeros([B, 2 * K, 2 * K])
    SNR_ = np.zeros([B])
    for i in range(B):
        # print i
        SNR = np.random.uniform(low=snr_low, high=snr_high)
        H = np.concatenate((np.concatenate((H_R[i, :, :], -1 * H_I[i, :, :]), axis=1),
                            np.concatenate((H_I[i, :, :], H_R[i, :, :]), axis=1)), axis=0)
        tmp_snr = (H.T.dot(H)).trace() / (2 * K)
        H = H / np.sqrt(tmp_snr) * np.sqrt(SNR)
        H_[i, :, :] = H
        y_[i, :] = H.dot(x_[i, :])  # +w[i,:]
        Hy_[i, :] = H.T.dot(y_[i, :])
        HH_[i, :, :] = H.T.dot(H_[i, :, :])
        SNR_[i] = SNR
    return y_,H_,Hy_,HH_,x_,SNR_, H_R, H_I, x_r, x_i, w_R, w_I,x_ind

def generate_data_iid_test(B,K,N,snr_low,snr_high):
    x_bit = np.random.randint(2, size=(B, 3 * K))
    x_r = np.zeros((B, K))
    x_i = np.zeros((B, K))
    x_ind = np.zeros((B, 8*K))
    for i in range(B):
        for ii in range(K):
            sym = x_bit[i, 3 * ii:3 * ii + 3]
            if sym[0] == 0 and sym[1] == 0 and sym[2] == 0:
                x_r[i, ii] = -1.0/np.sqrt(2)
                x_i[i, ii] = -1.0/np.sqrt(2)
                x_ind[i,0+ii*8] = 1
            if sym[0] == 0 and sym[1] == 0 and sym[2] == 1:
                x_r[i, ii] = -1
                x_i[i, ii] = 0
                x_ind[i,1+ii*8] = 1
            if sym[0] == 0 and sym[1] == 1 and sym[2] == 1:
                x_r[i, ii] = -1.0/np.sqrt(2)
                x_i[i, ii] = 1.0/np.sqrt(2)
                x_ind[i,2+ii*8] = 1
            if sym[0] == 0 and sym[1] == 1 and sym[2] == 0:
                x_r[i, ii] = 0
                x_i[i, ii] = 1
                x_ind[i,3+ii*8] = 1
            if sym[0] == 1 and sym[1] == 1 and sym[2] == 0:
                x_r[i, ii] = 1.0/np.sqrt(2)
                x_i[i, ii] = 1.0/np.sqrt(2)
                x_ind[i,4+ii*8] = 1
            if sym[0] == 1 and sym[1] == 1 and sym[2] == 1:
                x_r[i, ii] = 1
                x_i[i, ii] = 0
                x_ind[i,5+ii*8] = 1
            if sym[0] == 1 and sym[1] == 0 and sym[2] == 1:
                x_r[i, ii] = 1.0/np.sqrt(2)
                x_i[i, ii] = -1.0/np.sqrt(2)
                x_ind[i,6+ii*8] = 1
            if sym[0] == 1 and sym[1] == 0 and sym[2] == 0:
                x_r[i, ii] = 0
                x_i[i, ii] = -1
                x_ind[i,7+ii*8] = 1

    x_ = np.concatenate((x_r, x_i), axis=1)

    H_R = np.random.randn(B, N, K)
    H_I = np.random.randn(B, N, K)
    H_ = np.zeros([B, 2 * N, 2 * K])

    y_ = np.zeros([B, 2 * N])

    w_R = np.random.randn(B, N)
    w_I = np.random.randn(B, N)
    w = np.concatenate((w_R, w_I), axis=1)

    Hy_ = x_ * 0
    HH_ = np.zeros([B, 2 * K, 2 * K])
    SNR_ = np.zeros([B])
    for i in range(B):
        # print i
        SNR = np.random.uniform(low=snr_low, high=snr_high)
        H = np.concatenate((np.concatenate((H_R[i, :, :], -1 * H_I[i, :, :]), axis=1),
                            np.concatenate((H_I[i, :, :], H_R[i, :, :]), axis=1)), axis=0)
        tmp_snr = (H.T.dot(H)).trace() / (2 * K)
        H = H / np.sqrt(tmp_snr) * np.sqrt(SNR)
        H_[i, :, :] = H
        y_[i, :] = H.dot(x_[i, :])   +w[i,:]#*np.sqrt(tmp_snr)/np.sqrt(SNR)
        Hy_[i, :] = H.T.dot(y_[i, :])
        HH_[i, :, :] = H.T.dot(H_[i, :, :])
        SNR_[i] = SNR
    return y_,H_,Hy_,HH_,x_,SNR_, H_R, H_I, x_r, x_i, w_R, w_I,x_ind

def generate_data_train(B,K,N,snr_low,snr_high):
    x_bit = np.random.randint(2, size=(B, 3 * K))
    x_r = np.zeros((B, K))
    x_i = np.zeros((B, K))
    x_ind = np.zeros((B, 8*K))
    for i in range(B):
        for ii in range(K):
            sym = x_bit[i, 3 * ii:3 * ii + 3]
            if sym[0] == 0 and sym[1] == 0 and sym[2] == 0:
                x_r[i, ii] = -1.0/np.sqrt(2)
                x_i[i, ii] = -1.0/np.sqrt(2)
                x_ind[i,0+ii*8] = 1
            if sym[0] == 0 and sym[1] == 0 and sym[2] == 1:
                x_r[i, ii] = -1
                x_i[i, ii] = 0
                x_ind[i,1+ii*8] = 1
            if sym[0] == 0 and sym[1] == 1 and sym[2] == 1:
                x_r[i, ii] = -1.0/np.sqrt(2)
                x_i[i, ii] = 1.0/np.sqrt(2)
                x_ind[i,2+ii*8] = 1
            if sym[0] == 0 and sym[1] == 1 and sym[2] == 0:
                x_r[i, ii] = 0
                x_i[i, ii] = 1
                x_ind[i,3+ii*8] = 1
            if sym[0] == 1 and sym[1] == 1 and sym[2] == 0:
                x_r[i, ii] = 1.0/np.sqrt(2)
                x_i[i, ii] = 1.0/np.sqrt(2)
                x_ind[i,4+ii*8] = 1
            if sym[0] == 1 and sym[1] == 1 and sym[2] == 1:
                x_r[i, ii] = 1
                x_i[i, ii] = 0
                x_ind[i,5+ii*8] = 1
            if sym[0] == 1 and sym[1] == 0 and sym[2] == 1:
                x_r[i, ii] = 1.0/np.sqrt(2)
                x_i[i, ii] = -1.0/np.sqrt(2)
                x_ind[i,6+ii*8] = 1
            if sym[0] == 1 and sym[1] == 0 and sym[2] == 0:
                x_r[i, ii] = 0
                x_i[i, ii] = -1
                x_ind[i,7+ii*8] = 1

    x_ = np.concatenate((x_r, x_i), axis=1)

    H_R = np.random.randn(B, N, K)
    H_I = np.random.randn(B, N, K)
    H_ = np.zeros([B, 2 * N, 2 * K])

    y_ = np.zeros([B, 2 * N])

    w_R = np.random.randn(B, N)
    w_I = np.random.randn(B, N)
    w = np.concatenate((w_R, w_I), axis=1)

    Hy_ = x_ * 0
    HH_ = np.zeros([B, 2 * K, 2 * K])
    SNR_ = np.zeros([B])
    for i in range(B):
        # print i
        SNR = np.random.uniform(low=snr_low, high=snr_high)
        H = np.concatenate((np.concatenate((H_R[i, :, :], -1 * H_I[i, :, :]), axis=1),
                            np.concatenate((H_I[i, :, :], H_R[i, :, :]), axis=1)), axis=0)
        tmp_snr = (H.T.dot(H)).trace() / (2 * K)
        H = H / np.sqrt(tmp_snr) * np.sqrt(SNR)
        H_[i, :, :] = H
        y_[i, :] = H.dot(x_[i, :]) + w[i,:] #*np.sqrt(tmp_snr)/np.sqrt(SNR)
        Hy_[i, :] = H.T.dot(y_[i, :])
        HH_[i, :, :] = H.T.dot(H_[i, :, :])
        SNR_[i] = SNR
    return y_,H_,Hy_,HH_,x_,SNR_, H_R, H_I, x_r, x_i, w_R, w_I, x_ind

def find_nearest_mpsk(values):
    values = tf.pow(values,2)*tf.sign(values)
    values = values+1
    values = values*2
    values = tf.clip_by_value(values,0,4)
    values = tf.round(values)
    values = values / 2
    values = values - 1
    values = tf.sqrt(tf.abs(values))*tf.sign(values)
    return values

def piecewise_linear_soft_sign(x):
    t = tf.constant(0.1)
    y = -3+tf.nn.relu(x+2+t)/(tf.abs(t)+0.00001)-tf.nn.relu(x+2-t)/(tf.abs(t)+0.00001)+tf.nn.relu(x+t)/(tf.abs(t)+0.00001)-tf.nn.relu(x-t)/(tf.abs(t)+0.00001)+tf.nn.relu(x-2+t)/(tf.abs(t)+0.00001)-tf.nn.relu(x-2-t)/(tf.abs(t)+0.00001)
    return y



def affine_layer(x,input_size,output_size,Layer_num):
    W = tf.Variable(tf.random_normal([input_size, output_size], stddev=0.01))
    w = tf.Variable(tf.random_normal([1, output_size], stddev=0.01))
    y = tf.matmul(x, W)+w
    return y

def relu_layer(x,input_size,output_size,Layer_num):
    y = tf.nn.relu(affine_layer(x,input_size,output_size,Layer_num))
    return y

def sign_layer(x,input_size,output_size,Layer_num):
    #y = piecewise_linear_soft_sign(affine_layer(x,input_size,output_size,Layer_num))
    y = affine_layer(x,input_size,output_size,Layer_num)
    return y

#tensorflow placeholders, the input given to the model in order to train and test the network
HY = tf.placeholder(tf.float32,shape=[None,2*K])
X = tf.placeholder(tf.float32,shape=[None,2*K])
HH = tf.placeholder(tf.float32,shape=[None, 2*K , 2*K])
X_IND = tf.placeholder(tf.float32,shape=[None,8*K])


batch_size = tf.shape(HY)[0]
X_LS = tf.matmul(tf.expand_dims(HY,1),tf.matrix_inverse(HH))
X_LS= tf.squeeze(X_LS,1)
loss_LS = tf.reduce_mean(tf.square(X - X_LS))
ber_LS = tf.reduce_mean(tf.cast(are_not_equal_mpsk_tf(X,find_nearest_mpsk(X_LS)), tf.float32))
BER_LS1 = X
BER_LS2 = find_nearest_mpsk(X_LS)
BER_LS3 = are_not_equal_mpsk_tf(BER_LS1, BER_LS2)
BER_LS4 = tf.reshape(BER_LS1, [batch_size, K, 2])
BER_LS5 = tf.reshape(BER_LS2, [batch_size, K, 2])
BER_LS6 = are_not_equal_mpsk_tf(BER_LS4, BER_LS5)
BER_LS7 = tf.reduce_sum(tf.cast(BER_LS6, tf.float32), 2)
BER_LS8 = tf.cast(tf.greater(BER_LS7, 0), tf.float32)
BER_LS9 = tf.reduce_sum(BER_LS8)

S1=[]
S1.append(tf.zeros([batch_size,2*K]))
S2=[]
S2.append(tf.zeros([batch_size,8*K]))
V=[]
V.append(tf.zeros([batch_size,v_size]))
LOSS=[]
LOSS.append(tf.zeros([]))
BER=[]
BER.append(tf.zeros([]))
delta = tf.Variable(tf.zeros(L*2,1))


for i in range(1, L):
    print('aaa')
    print(i)
    temp1 = tf.matmul(tf.expand_dims(S1[-1], 1), HH)
    temp1 = tf.squeeze(temp1, 1)
    
    Z1 = S1[-1] - delta[(i - 1) * 2] * HY + delta[(i - 1) * 2 + 1] * temp1
    Z = tf.concat([Z1, V[-1]], 1)
    ZZ = relu_layer(Z,(2*K) + v_size , hl_size,'relu'+str(i))
    
    S2.append(sign_layer(ZZ, hl_size, 8 * K, 'sign' + str(i)))
    S2[i] = (1 - res_alpha) * S2[i] + res_alpha * S2[i - 1]
    V.append(affine_layer(ZZ, hl_size, v_size, 'aff' + str(i)))
    V[i] = (1 - res_alpha) * V[i] + res_alpha * V[i - 1]
    temp_pos = tf.strided_slice(S2[i][:][:], [0, 0], [batch_size, 2 * K], [1, 2])
    temp_neg = tf.strided_slice(tf.reverse(S2[i][:][:], [-1]), [0, 0], [batch_size, 2 * K], [1, 2])
    temp_neg = tf.reverse(temp_neg, [-1])
    temp_0 = tf.strided_slice(S2[i][:][:], [0, 0], [batch_size, 8 * K], [1, 8])
    temp_1 = tf.strided_slice(S2[i][:][:], [0, 1], [batch_size, 8 * K], [1, 8])
    temp_2 = tf.strided_slice(S2[i][:][:], [0, 2], [batch_size, 8 * K], [1, 8])
    temp_3 = tf.strided_slice(S2[i][:][:], [0, 3], [batch_size, 8 * K], [1, 8])
    temp_4 = tf.strided_slice(S2[i][:][:], [0, 4], [batch_size, 8 * K], [1, 8])
    temp_5 = tf.strided_slice(S2[i][:][:], [0, 5], [batch_size, 8 * K], [1, 8])
    temp_6 = tf.strided_slice(S2[i][:][:], [0, 6], [batch_size, 8 * K], [1, 8])
    temp_7 = tf.strided_slice(S2[i][:][:], [0, 7], [batch_size, 8 * K], [1, 8])

    S1_real = (-1/tf.sqrt(2.0))*temp_0  +\
              (-1)*temp_1 + \
              (-1 / tf.sqrt(2.0))*temp_2 +\
                0 * temp_3 + \
              (1.0 / np.sqrt(2.0)) * temp_4 +\
              1 * temp_5 + \
              (1.0 / np.sqrt(2.0)) * temp_6 + \
              0 * temp_7

    S1_im = (-1 / tf.sqrt(2.0)) * temp_0 + \
              (0) * temp_1 + \
              (1 / tf.sqrt(2.0)) * temp_2 + \
              1 * temp_3 + \
              (1 / tf.sqrt(2.0)) * temp_4 + \
              0 * temp_5 + \
              (-1 / tf.sqrt(2.0)) * temp_6 + \
               (-1) * temp_7
    S1.append(tf.concat([S1_real, S1_im], 1))

    if LOG_LOSS == 1:
        LOSS.append(np.log(i) * tf.reduce_mean(tf.reduce_mean(tf.square(X_IND - S2[-1]), 1) ))#/ tf.reduce_mean(tf.square(X - X_LS), 1)))
    else:
        LOSS.append(tf.reduce_mean(tf.reduce_mean(tf.square(X_IND - S2[-1]), 1)))# / tf.reduce_mean(tf.square(X - X_LS), 1)))
    BER.append(tf.reduce_mean(tf.cast(tf.not_equal(X_IND, tf.round(S2[-1])), tf.float32)))
BER1 = tf.reshape(X_IND,[batch_size,K,8])
BER2 = tf.reshape(S2[-1], [batch_size, K, 8])
Max_Val = tf.reduce_max(BER2,axis=2, keep_dims=True)
Greater = tf.greater_equal(BER2,Max_Val)
BER3 = tf.round(tf.cast(Greater,tf.float32))
BER4 = tf.not_equal(BER1, BER3)
BER5 = tf.reduce_sum(tf.cast(BER4,tf.float32),2)
BER6 = tf.cast(tf.greater(BER5,0),tf.float32)
SER =  tf.reduce_sum(BER6)  
TOTAL_LOSS=tf.add_n(LOSS)


global_step1 = tf.Variable(0, trainable=False)
learning_rate1 = tf.train.exponential_decay(startingLearningRate1, global_step1, decay_step_size1, decay_factor1, staircase=True)
train_step1 = tf.train.AdamOptimizer(learning_rate1).minimize(TOTAL_LOSS)

global_step2 = tf.Variable(0, trainable=False)
learning_rate2 = tf.train.exponential_decay(startingLearningRate2, global_step2, decay_step_size2, decay_factor2, staircase=True)
train_step2 = tf.train.AdamOptimizer(learning_rate2).minimize(TOTAL_LOSS)

init_op=tf.initialize_all_variables()

sess.run(init_op)
#Training DetNet
for i in range(train_iter_no_noise): #num of train iter
    batch_Y, batch_H, batch_HY, batch_HH, batch_X , SNR1, H_R, H_I, x_R, x_I, w_R, w_I,x_ind= generate_data_train_no_noise(train_batch_size,K,N,snr_low,snr_high)
    train_step1.run(feed_dict={HY: batch_HY, HH: batch_HH, X: batch_X,X_IND:x_ind})
    if i % 1000 == 0 :
        sys.stderr.write(str(i)+' ')
        batch_Y, batch_H, batch_HY, batch_HH, batch_X ,SNR1, H_R, H_I, x_R, x_I, w_R, w_I,x_ind= generate_data_iid_test_no_noise(train_batch_size,K,N,snr_low,snr_high)
        results = sess.run([loss_LS,LOSS[L-1],ber_LS,SER], {HY: batch_HY, HH: batch_HH, X: batch_X,X_IND:x_ind})
        print_string = [i]+results
        print ' '.join('%s' % x for x in print_string)


for i in range(train_iter): #num of train iter
    batch_Y, batch_H, batch_HY, batch_HH, batch_X , SNR1, H_R, H_I, x_R, x_I, w_R, w_I,x_ind= generate_data_train(train_batch_size,K,N,snr_low,snr_high)
    train_step2.run(feed_dict={HY: batch_HY, HH: batch_HH, X: batch_X,X_IND:x_ind})

    if i % 1000 == 0 :
	sys.stderr.write(str(i)+ ' ')
        batch_Y, batch_H, batch_HY, batch_HH, batch_X ,SNR1, H_R, H_I, x_R, x_I, w_R, w_I,x_ind= generate_data_iid_test(train_batch_size,K,N,snr_low,snr_high)
        results = sess.run([loss_LS,LOSS[L-1],ber_LS,SER], {HY: batch_HY, HH: batch_HH, X: batch_X,X_IND:x_ind})

        print_string = [i]+results
        print ' '.join('%s' % x for x in print_string)

#Testing the trained model
snrdb_list = np.linspace(snrdb_low_test,snrdb_high_test,num_snr)
snr_list = 10.0 ** (snrdb_list/10.0)
bers = np.zeros((4,num_snr))
sers = np.zeros((4,num_snr))
times = np.zeros((4,num_snr))
tmp_bers = np.zeros((4,test_iter))
tmp_sers = np.zeros((4,test_iter))

tmp_times = np.zeros((4,test_iter))
for j in range(num_snr):
    for jj in range(test_iter):
        sys.stderr.write(str(jj) + ' ')
        print('snr:')
        print(snrdb_list[j])
        print('test iteration:')
        print(jj)

        batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1, H_R, H_I, x_R, x_I, w_R, w_I,x_ind= generate_data_iid_test(test_batch_size , K,N,snr_list[j],snr_list[j])
        tic = tm.time()
        tmp_bers[2,jj] = np.array(sess.run(SER, {HY: batch_HY, HH: batch_HH, X: batch_X,X_IND:x_ind}))/(test_batch_size*K)
        toc = tm.time()
        tmp_times[2][jj] =toc - tic
        tmp_bers[0,jj] = np.array(sess.run(BER_LS9, {HY: batch_HY, HH: batch_HH, X: batch_X}))/(K*test_batch_size)
        tmp_bers[3,jj],tmp_sers[3,jj] = batch_sdr2(batch_Y,batch_H,batch_X)

    bers[:,j] = np.mean(tmp_bers,1)
    sers[:,j] = np.mean(tmp_sers,1)

    times[:,j] = np.mean(tmp_times[0])/test_batch_size

print('snrdb_list')
print(snrdb_list)
print('bers')
print(bers)
print('sers')
print(sers)
print('times')
print(times)


def sphdec_core(z, R, symbset, layer,dist):
    global SPHDEC_RADIUS
    global RETVAL
    global TMPVAL
    global SYMBSETSIZE
    global SEARCHFLAG
    if (layer == 0):
        for ii in range(SYMBSETSIZE):
            TMPVAL[0] = deepcopy(symbset[ii])
            #print('R')
            #print(R)
            #print('TMPVAL')
            #print(TMPVAL)
            d = np.power(np.abs(z[0] - np.dot(R[0,:],TMPVAL)) , 2) + dist
            if (d <= SPHDEC_RADIUS):
                RETVAL = deepcopy(TMPVAL)
                SPHDEC_RADIUS = deepcopy(d)

                SEARCHFLAG = SEARCHFLAG + 1
    else:
        for jj in range(SYMBSETSIZE):
            TMPVAL[layer] = deepcopy(symbset[jj])
            #print('R')
            #print(R)
            #print('TMPVAL')
            #print(TMPVAL)
            d = np.power(np.abs(z[layer] - np.dot(R[layer][layer:] ,TMPVAL[layer:])),2) + dist
            if (d <= SPHDEC_RADIUS):
                sphdec_core(z, R, symbset, layer-1, d)


def sphdec(H, y, symbset, radius,true_x):
    global RETVAL
    global TMPVAL
    global SYMBSETSIZE
    global SEARCHFLAG
    global SPHDEC_RADIUS
    Q,R = np.linalg.qr(H,mode='reduced')
    z = np.dot(np.conj(Q).T,y)

    K = np.shape(H)[1]

    RETVAL = np.zeros((K, 1),dtype=complex)
    TMPVAL = np.zeros((K, 1),dtype=complex)
    SYMBSETSIZE = len(symbset)
    SEARCHFLAG = 0
    SPHDEC_RADIUS= radius
    sphdec_core(z, R, symbset, K-1, 0)
    if SEARCHFLAG > 0:
        r = RETVAL
    else:
        r = np.ones((K,1))

    return r

def batch_sd(K,N,constellation, batch_Y, batch_H, batch_X, test_batch_size,x_R, x_I, H_R, H_I,radius):
    SER = 0
    for iter in range(test_batch_size):
        y_R_iter = batch_Y[iter,0:N]
        y_I_iter = batch_Y[iter,N:2 * N]
        x_R_iter = x_R[iter]
        x_I_iter = x_I[iter]
        H_R_iter = batch_H[iter,0:N,0:K]
        H_I_iter = batch_H[iter,N:2*N,0:K]
        y = y_R_iter + 1j*y_I_iter
        H = H_R_iter + 1j*H_I_iter
        x = x_R_iter + 1j*x_I_iter
        x_sd = sphdec(H,y,constellation,radius,x)
        temp_ser =np.sum(are_not_equal_mpsk_np(np.transpose(x_sd),np.transpose(x)))

        SER = SER+temp_ser
    SER = SER/(1.0*K*test_batch_size)

    return SER

constellation = [-1, 1, 1j, -1j, 1/np.sqrt(2)+(1j/np.sqrt(2)), -1/np.sqrt(2)+(1j/np.sqrt(2)), 1/np.sqrt(2)-(1j/np.sqrt(2)), -1/np.sqrt(2)-(1j/np.sqrt(2))]

test_iter= [2000,4000,10000,20000,40000,80000]
max_radius=(N*2)*1.5
for i in range(4):
    sys.stderr.write(str(i) + ' ')
    max_radius = max_radius*1.15
    BERS = np.zeros([num_snr,1])
    SERS = np.zeros([num_snr, 1])
    Times = np.zeros([num_snr,1])
    for j in range(num_snr):
        sys.stderr.write(str(j) + 'num_snr ')
        temp_times= np.zeros([test_iter[j],1])
        temp_ber = 0
        temp_ser = 0
        for jj in range(test_iter[j]):
            batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1, H_R, H_I, x_R, x_I, w_R, w_I,x_ind= generate_data_iid_test(1 , K,N,snr_list[j],snr_list[j]) #in sd batch size is 1

            tic = tm.time()
            temp_ser = temp_ser + np.float32(batch_sd(K,N,constellation, batch_Y, batch_H, batch_X, 1, x_R, x_I, H_R, H_I, max_radius))
            toc = tm.time()
            temp_times[jj] = (toc-tic)

        Times[j] = np.mean(temp_times)
        
        SERS[j] = temp_ser/test_iter[j]
        print('SERS[j]')
        print(SERS[j])
    print('i')
    print(i)
    print('max_radius')
    print(max_radius)
    print('BERS')
    print(BERS)
    print('SERS')
    print(SERS)
    print('Times')
    print(Times)