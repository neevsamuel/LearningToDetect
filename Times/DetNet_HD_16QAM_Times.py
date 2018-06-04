#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import time as tm
import math
import sys
import pickle as pkl
from copy import deepcopy

def find_nearest_np(values):
    values = values + 3
    values = values/2
    values = np.clip(values,0,3)
    values = np.round(values)
    values = values * 2
    values = values - 3
    return values

def gaus(s,mean,var):
    s = np.array(s)
    return np.exp(-(np.power((s-mean),2)/(2*np.power(var,2))))

def ampF13(s,tau):
    num =  gaus(s,1,tau) + (-1)*gaus(s,-1,tau) + 3*gaus(s,3,tau) + (-3)*gaus(s,-3,tau)
    denum = gaus(s,1,tau) + gaus(s,-1,tau) + gaus(s,3,tau) + gaus(s,-3,tau)
    return num/(np.abs(denum) + 0.00001)

def ampG13(s,tau):
    num =  gaus(s,1,tau) + gaus(s,-1,tau) + 9*gaus(s,3,tau) + 9*gaus(s,-3,tau)
    denum = gaus(s,1,tau) + gaus(s,-1,tau) + gaus(s,3,tau) + gaus(s,-3,tau)
    second  = np.power(np.abs(ampF13(s,tau)),2)
    return num/(np.abs(denum) + 0.00001) - second

def round13(s, K):
    retVal =  np.zeros(K)
    vals = np.array([-3 ,-1 ,1 ,3])
    for i in range(K):
        retVal[i] = vals[(np.abs(vals-s[i])).argmin()]
    return retVal

def round13_2(s, K, B):
    retVal =  np.zeros((B,K))
    vals = np.array([-3 ,-1 ,1 ,3])
    for i in range(B):
        for ii in range(K):
            retVal[i][ii] = vals[(np.abs(vals-s[i][ii])).argmin()]
    return retVal

def amp13(y,H,N0,N,K):

    L = K*3
    beta = K/(0.+N)
    s = np.zeros(2*K)
    tau = beta*1/N0
    r=y
    for it in range(L):
        z = s+np.dot(H.T,r)
        s = ampF13(z,N0*(1+tau))
        tau_new = beta/N0*np.mean(ampG13(z,N0*(1+tau)))
        r = y - np.dot(H,s)+tau_new/(1+tau)*r
        tau = tau_new
    return round13(s,2*K)

def amp13_2(y,H,N0,K,N,B):
    
    L = K*3
    beta = np.ones((B,1))*K/(0.+N)
    s = np.zeros((B,2*K,1))
    tau = beta*1/N0
    tau = np.expand_dims(tau,axis=2)
    y =np.expand_dims(y,axis=2)
    r=y
    for it in range(L):
        z = s+np.matmul(np.transpose(H,(0,2,1)),r)
        s = ampF13(z,N0*(1.0+tau))
        tau_new = beta/N0*np.mean(ampG13(z,N0*(1.0+tau)),1)
        tau_new = np.expand_dims(tau_new,axis=2)
        r = y - np.matmul(H,s)+tau_new/(1.0+tau)*r
        tau = tau_new
    return round13_2(s,2*K,B)




def batch_amp13(N,K,batch_Y,batch_H,batch_X,n0,B,SNR, x_R, x_I):
    err_amp = 0.0
    for i in range(B):
        xx = amp13(batch_Y[i]/np.sqrt(SNR),batch_H[i]/np.sqrt(SNR),n0,N,K)
        retValReal = xx[0:K]
        retValIm = xx[K:2 * K]
        err_amp += (np.mean(np.logical_or(np.not_equal(x_R[i], retValReal), np.not_equal(x_I[i], retValIm)))) / (B)
    return err_amp

def batch_amp13_2(N,K,batch_Y,batch_H,batch_X,n0,B,SNR, x_R, x_I):
    err_amp = 0.0
    xx = amp13_2(batch_Y/np.sqrt(SNR),batch_H/np.sqrt(SNR),n0,K,N,B)
    if B > 1:
        xx =np.squeeze(xx)

    retValReal = xx[:,0:K]
    retValIm = xx[:,K:2 * K]
    err_amp += np.mean(np.mean(np.logical_or(np.not_equal(x_R, retValReal), np.not_equal(x_I, retValIm))))
    return err_amp

def dfe(y,H):
    Q,R = np.linalg.qr(H)
    Qy = np.dot (Q.T,y)
    xx=np.zeros([2*K])
    for k in range(2*K-1,-1,-1):
        xx[k]=find_nearest_np((Qy[k]-np.dot(R[k][k:],xx[k:]))/R[k][k])
    return(xx)

def batch_dfe(y,H,x,x_R,x_I):
    B = np.shape(y)[0]
    ber=0
    for i in range(B):
        xx = dfe(y[i].T,H[i])
        retValReal = xx[0:K]
        retValIm = xx[K:2 * K]
        ber+=(np.mean(np.logical_or(np.not_equal(x_R[i], retValReal), np.not_equal(x_I[i], retValIm))))
    ber=ber/B
    return np.float32(ber)


###start here

sess = tf.InteractiveSession()

#parameters
K = 15
N = 25
snrdb_low = 7.0
snrdb_high = 14.0
snr_low = 10.0 ** (snrdb_low/10.0)
snr_high = 10.0 ** (snrdb_high/10.0)
L=30
v_size = 4*(2*K)
hl_size = 8*(2*K)
startingLearningRate1 = 0.0003
startingLearningRate2 = 0.0003
decay_factor1 = 0.97
decay_factor2 = 0.97
decay_step_size1 = 1000
decay_step_size2 = 1000
train_iter = 1
train_iter_no_noise = 1
n0 = 0.5

train_batch_size = 2

test_iter= 10

test_batch_size = 10

LOG_LOSS = 1
res_alpha=0.9
num_snr = 6
snrdb_low_test=8.0
snrdb_high_test=13.0
symbols = np.array([-3,-1,1,3])

print('16QAM one hot times')
print(K)
print(N)
print(snrdb_low)
print(snrdb_high)
print(snr_low)
print(snr_high)
print('L')
print(L)
print('v_size')
print(v_size)
print('hl_size')

print(hl_size)
print(startingLearningRate1)
print(startingLearningRate2)
print(decay_factor1)
print(decay_factor2)
print(decay_step_size1)
print(decay_step_size2)
print(train_iter)
print(train_batch_size)
print('test_iter')
print(test_iter)
print('test_batch_size')
print(test_batch_size)
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
    x_R = np.random.randint(4,size = (B,K))
    x_R = x_R * 2
    x_R = x_R - 3

    x_I = np.random.randint(4, size=(B, K))
    x_I = x_I * 2
    x_I = x_I - 3

    x_ind = np.zeros([B,K,16])
    for i in range(B):
        for ii in range(K):
            if x_R[i,ii]==-3 and x_I[i,ii] == -3:
                x_ind[i,ii,0] =  1
            if x_R[i,ii]==-3 and x_I[i,ii] == -1:
                x_ind[i,ii,1] =  1
            if x_R[i,ii]==-3 and x_I[i,ii] == 1:
                x_ind[i,ii,2] =  1
            if x_R[i,ii]==-3 and x_I[i,ii] == 3:
                x_ind[i,ii,3] =  1
            if x_R[i,ii]==-1 and x_I[i,ii] == -3:
                x_ind[i,ii,4] =  1
            if x_R[i,ii]==-1 and x_I[i,ii] == -1:
                x_ind[i,ii,5] =  1
            if x_R[i,ii]==-1 and x_I[i,ii] == 1:
                x_ind[i,ii,6] =  1
            if x_R[i,ii]==-1 and x_I[i,ii] == 3:
                x_ind[i,ii,7] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == -3:
                x_ind[i,ii,8] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == -1:
                x_ind[i,ii,9] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == 1:
                x_ind[i,ii,10] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == 3:
                x_ind[i,ii,11] =  1
            if x_R[i,ii]==3 and x_I[i,ii] == -3:
                x_ind[i,ii,12] =  1
            if x_R[i,ii]==3 and x_I[i,ii] == -1:
                x_ind[i,ii,13] =  1
            if x_R[i,ii]==3 and x_I[i,ii] == 1:
                x_ind[i,ii,14] =  1
            if x_R[i,ii]==3 and x_I[i,ii] == 3:
                x_ind[i,ii,15] =  1

    x_  = np.concatenate((x_R , x_I) , axis = 1)

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
    return y_,H_,Hy_,HH_,x_,SNR_, H_R, H_I, x_R, x_I, w_R, w_I, x_ind


def generate_data_train_no_noise(B,K,N,snr_low,snr_high):
    x_R = np.random.randint(4, size=(B, K))
    x_R = x_R * 2
    x_R = x_R - 3

    x_I = np.random.randint(4, size=(B, K))
    x_I = x_I * 2
    x_I = x_I - 3
    
    x_ind = np.zeros([B,K,16])
    for i in range(B):
        for ii in range(K):
            if x_R[i,ii]==-3 and x_I[i,ii] == -3:
                x_ind[i,ii,0] =  1
            if x_R[i,ii]==-3 and x_I[i,ii] == -1:
                x_ind[i,ii,1] =  1
            if x_R[i,ii]==-3 and x_I[i,ii] == 1:
                x_ind[i,ii,2] =  1
            if x_R[i,ii]==-3 and x_I[i,ii] == 3:
                x_ind[i,ii,3] =  1
            if x_R[i,ii]==-1 and x_I[i,ii] == -3:
                x_ind[i,ii,4] =  1
            if x_R[i,ii]==-1 and x_I[i,ii] == -1:
                x_ind[i,ii,5] =  1
            if x_R[i,ii]==-1 and x_I[i,ii] == 1:
                x_ind[i,ii,6] =  1
            if x_R[i,ii]==-1 and x_I[i,ii] == 3:
                x_ind[i,ii,7] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == -3:
                x_ind[i,ii,8] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == -1:
                x_ind[i,ii,9] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == 1:
                x_ind[i,ii,10] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == 3:
                x_ind[i,ii,11] =  1
            if x_R[i,ii]==3 and x_I[i,ii] == -3:
                x_ind[i,ii,12] =  1
            if x_R[i,ii]==3 and x_I[i,ii] == -1:
                x_ind[i,ii,13] =  1
            if x_R[i,ii]==3 and x_I[i,ii] == 1:
                x_ind[i,ii,14] =  1
            if x_R[i,ii]==3 and x_I[i,ii] == 3:
                x_ind[i,ii,15] =  1

    x_ = np.concatenate((x_R, x_I), axis=1)

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
    return y_,H_,Hy_,HH_,x_,SNR_, H_R, H_I, x_R, x_I, w_R, w_I,x_ind

def generate_data_iid_test(B,K,N,snr_low,snr_high):
    x_R = np.random.randint(4, size=(B, K))
    x_R = x_R * 2
    x_R = x_R - 3
    
    

    x_I = np.random.randint(4, size=(B, K))
    x_I = x_I * 2
    x_I = x_I - 3
    
    x_ind = np.zeros([B,K,16])
    for i in range(B):
        for ii in range(K):
            if x_R[i,ii]==-3 and x_I[i,ii] == -3:
                x_ind[i,ii,0] =  1
            if x_R[i,ii]==-3 and x_I[i,ii] == -1:
                x_ind[i,ii,1] =  1
            if x_R[i,ii]==-3 and x_I[i,ii] == 1:
                x_ind[i,ii,2] =  1
            if x_R[i,ii]==-3 and x_I[i,ii] == 3:
                x_ind[i,ii,3] =  1
            if x_R[i,ii]==-1 and x_I[i,ii] == -3:
                x_ind[i,ii,4] =  1
            if x_R[i,ii]==-1 and x_I[i,ii] == -1:
                x_ind[i,ii,5] =  1
            if x_R[i,ii]==-1 and x_I[i,ii] == 1:
                x_ind[i,ii,6] =  1
            if x_R[i,ii]==-1 and x_I[i,ii] == 3:
                x_ind[i,ii,7] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == -3:
                x_ind[i,ii,8] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == -1:
                x_ind[i,ii,9] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == 1:
                x_ind[i,ii,10] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == 3:
                x_ind[i,ii,11] =  1
            if x_R[i,ii]==3 and x_I[i,ii] == -3:
                x_ind[i,ii,12] =  1
            if x_R[i,ii]==3 and x_I[i,ii] == -1:
                x_ind[i,ii,13] =  1
            if x_R[i,ii]==3 and x_I[i,ii] == 1:
                x_ind[i,ii,14] =  1
            if x_R[i,ii]==3 and x_I[i,ii] == 3:
                x_ind[i,ii,15] =  1
                     
    x_ = np.concatenate((x_R, x_I), axis=1)

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
        y_[i, :] = H.dot(x_[i, :])   +w[i,:]
        Hy_[i, :] = H.T.dot(y_[i, :])
        HH_[i, :, :] = H.T.dot(H_[i, :, :])
        SNR_[i] = SNR
    return y_,H_,Hy_,HH_,x_,SNR_, H_R, H_I, x_R, x_I, w_R, w_I,x_ind

def generate_data_train(B,K,N,snr_low,snr_high):
    x_R = np.random.randint(4, size=(B, K))
    x_R = x_R * 2
    x_R = x_R - 3

    x_I = np.random.randint(4, size=(B, K))
    x_I = x_I * 2
    x_I = x_I - 3

    x_ind = np.zeros([B,K,16])
    for i in range(B):
        for ii in range(K):
            if x_R[i,ii]==-3 and x_I[i,ii] == -3:
                x_ind[i,ii,0] =  1
            if x_R[i,ii]==-3 and x_I[i,ii] == -1:
                x_ind[i,ii,1] =  1
            if x_R[i,ii]==-3 and x_I[i,ii] == 1:
                x_ind[i,ii,2] =  1
            if x_R[i,ii]==-3 and x_I[i,ii] == 3:
                x_ind[i,ii,3] =  1
            if x_R[i,ii]==-1 and x_I[i,ii] == -3:
                x_ind[i,ii,4] =  1
            if x_R[i,ii]==-1 and x_I[i,ii] == -1:
                x_ind[i,ii,5] =  1
            if x_R[i,ii]==-1 and x_I[i,ii] == 1:
                x_ind[i,ii,6] =  1
            if x_R[i,ii]==-1 and x_I[i,ii] == 3:
                x_ind[i,ii,7] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == -3:
                x_ind[i,ii,8] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == -1:
                x_ind[i,ii,9] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == 1:
                x_ind[i,ii,10] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == 3:
                x_ind[i,ii,11] =  1
            if x_R[i,ii]==3 and x_I[i,ii] == -3:
                x_ind[i,ii,12] =  1
            if x_R[i,ii]==3 and x_I[i,ii] == -1:
                x_ind[i,ii,13] =  1
            if x_R[i,ii]==3 and x_I[i,ii] == 1:
                x_ind[i,ii,14] =  1
            if x_R[i,ii]==3 and x_I[i,ii] == 3:
                x_ind[i,ii,15] =  1

    x_ = np.concatenate((x_R, x_I), axis=1)

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
        y_[i, :] = H.dot(x_[i, :])   +w[i,:]
        Hy_[i, :] = H.T.dot(y_[i, :])
        HH_[i, :, :] = H.T.dot(H_[i, :, :])
        SNR_[i] = SNR
    return y_,H_,Hy_,HH_,x_,SNR_, H_R, H_I, x_R, x_I, w_R, w_I,x_ind

def find_nearest(values):
    values = values + 3
    values = values/2
    values = tf.clip_by_value(values,0,3)
    values = tf.round(values)
    values = values * 2
    values = values - 3
    return values

def piecewise_linear_soft_sign(x):
    #t = tf.Variable(0.1)
    t = tf.constant(0.1)
    y = -3+tf.nn.relu(x+2+t)/(tf.abs(t)+0.00001)-tf.nn.relu(x+2-t)/(tf.abs(t)+0.00001)+tf.nn.relu(x+t)/(tf.abs(t)+0.00001)-tf.nn.relu(x-t)/(tf.abs(t)+0.00001)+tf.nn.relu(x-2+t)/(tf.abs(t)+0.00001)-tf.nn.relu(x-2-t)/(tf.abs(t)+0.00001)
    return y

#def piecewise_linear_soft_sign(x):
#    t = tf.Variable(0.1)
#    y = -1+tf.nn.relu(x+t)/(tf.abs(t)+0.00001)-tf.nn.relu(x-t)/(tf.abs(t)+0.00001)
#    return y

def affine_layer(x,input_size,output_size,Layer_num):
    W = tf.Variable(tf.random_normal([input_size, output_size], stddev=0.01))
    w = tf.Variable(tf.random_normal([1, output_size], stddev=0.01))
    y = tf.matmul(x, W)+w
    return y,W,w

def relu_layer(x,input_size,output_size,Layer_num):
    y,W,w = affine_layer(x,input_size,output_size,Layer_num)
    y= tf.nn.relu(y)
    return y,W,w

def sign_layer(x,input_size,output_size,Layer_num):
    #y = piecewise_linear_soft_sign(affine_layer(x,input_size,output_size,Layer_num))
    y,W,w= affine_layer(x,input_size,output_size,Layer_num)
    return y,W,w

#tensorflow placeholders, the input given to the model in order to train and test the network
HY = tf.placeholder(tf.float32,shape=[None,2*K])
X = tf.placeholder(tf.float32,shape=[None,2*K])
HH = tf.placeholder(tf.float32,shape=[None, 2*K , 2*K])
X_IND = tf.placeholder(tf.float32,shape=[None,K,16])


batch_size = tf.shape(HY)[0]
X_LS = tf.matmul(tf.expand_dims(HY,1),tf.matrix_inverse(HH))
X_LS= tf.squeeze(X_LS,1)
loss_LS = tf.reduce_mean(tf.square(X - X_LS))
xLS_real = find_nearest(X_LS)[:,0:K]
xLS_imag = find_nearest(X_LS)[:,K:2*K]
x_real = X[:,0:K]
x_imag = X[:,K:2*K]
ber_LS =  tf.reduce_mean(tf.cast(tf.logical_or(tf.not_equal(x_real,xLS_real) , tf.not_equal(x_imag,xLS_imag)), tf.float32))


S1=[]
S1.append(tf.zeros([batch_size,2*K]))
S2=[]
S2.append(tf.zeros([batch_size,16*K]))
V=[]
V.append(tf.zeros([batch_size,v_size]))
LOSS=[]
LOSS.append(tf.zeros([]))
BER=[]
BER.append(tf.zeros([]))
delta = tf.Variable(tf.zeros(L*2,1))
Z1 = []
ZZ1 = []
W11 = []
w11 = []
W22 = []
w22 = []
W33 = []
w33 = []
Z1_1 = []
temp1 = []
Z = []
first = []
second = []
X_IND_RESHAPED = tf.reshape(X_IND,[batch_size,2*K])
#The architecture of DetNet
for i in range(1,L):
    print('i')
    print(i)
    temp11 = tf.matmul(tf.expand_dims(S1[-1],1),HH)
    temp1.append(tf.squeeze(temp11,1))
    first.append(delta[(i-1) * 2]*HY)
    second.append(delta[(i-1) * 2 + 1]*temp1[-1])
    Z1.append(S1[-1] - (delta[(i-1) * 2]*HY) + (delta[(i-1) * 2 + 1]*temp1[-1]))
    Z.append(tf.concat([Z1[-1], V[-1]], 1))
    ZZ,Wtemp,wtemp = relu_layer(Z[-1],(2*K) + v_size , hl_size,'relu'+str(i))
    
    W11.append(Wtemp)
    w11.append(wtemp)
    ZZ1.append((ZZ))
    
    S2_temp,W22_temp,w22_temp = sign_layer(ZZ , hl_size , 16*K,'sign'+str(i))
    
    S2.append(S2_temp)
    W22.append(W22_temp)
    w22.append(w22_temp)
    
    S2[i]=(1-res_alpha)*S2[i]+res_alpha*S2[i-1]
    S2[i] =    tf.clip_by_value(S2[i],0,1)

    V_temp ,  W33_temp , w33_temp = affine_layer(ZZ , hl_size , v_size,'aff'+str(i))
    
    V.append(V_temp)
    W33.append(W33_temp)
    w33.append(w33_temp)
    
    V[i]=(1-res_alpha)*V[i]+res_alpha*V[i-1] 
    
    S3 = tf.reshape(S2[i],[batch_size,K,16])

    temp_0 = S3[:,:,0]
    temp_1 = S3[:,:,1]
    temp_2 = S3[:,:,2]
    temp_3 = S3[:,:,3]
    temp_4 = S3[:,:,4]
    temp_5 = S3[:,:,5]
    temp_6 = S3[:,:,6]
    temp_7 = S3[:,:,7]
    temp_8 = S3[:,:,8]
    temp_9 = S3[:,:,9]
    temp_10 = S3[:,:,10]
    temp_11 = S3[:,:,11]
    temp_12 = S3[:,:,12]
    temp_13 = S3[:,:,13]
    temp_14 = S3[:,:,14]    
    temp_15 = S3[:,:,15]
    
    S1_real = -3.0*temp_0  +\
              -3.0*temp_1  +\
              -3.0*temp_2  +\
              -3.0*temp_3  +\
              -1.0*temp_4  +\
              -1.0*temp_5  +\
              -1.0*temp_6  +\
              -1.0*temp_7  +\
               1.0*temp_8  +\
               1.0*temp_9  +\
               1.0*temp_10 +\
               1.0*temp_11 +\
               3.0*temp_12 +\
               3.0*temp_13 +\
               3.0*temp_14 +\
               3.0*temp_15

    S1_im =   -3.0*temp_0  +\
              -1.0*temp_1  +\
               1.0*temp_2  +\
               3.0*temp_3  +\
              -3.0*temp_4  +\
              -1.0*temp_5  +\
               1.0*temp_6  +\
               3.0*temp_7  +\
              -3.0*temp_8  +\
              -1.0*temp_9  +\
               1.0*temp_10 +\
               3.0*temp_11 +\
              -3.0*temp_12 +\
              -1.0*temp_13 +\
               1.0*temp_14 +\
               3.0*temp_15
               
    S1.append(tf.concat([S1_real, S1_im], 1))
              

    X_IND_reshaped = tf.reshape(X_IND,[batch_size,16*K])
    if LOG_LOSS == 1:
        LOSS.append(np.log(i)*tf.reduce_mean(tf.reduce_mean(tf.square(X_IND_reshaped - S2[-1]),1)))#/tf.reduce_mean(tf.square(X - X_LS),1)))
    else:
        LOSS.append(tf.reduce_mean(tf.reduce_mean(tf.square(X_IND_reshaped - S2[-1]),1)))#/tf.reduce_mean(tf.square(X - X_LS),1)))
    BER.append(tf.reduce_mean(tf.cast(tf.not_equal(X_IND, tf.round(S3)), tf.float32)))
Max_Val = tf.reduce_max(S3,axis=2, keep_dims=True)
Greater = tf.greater_equal(S3,Max_Val)
BER2 = tf.round(tf.cast(Greater,tf.float32))
#BER2 = tf.round(S3) 
BER3 = tf.not_equal(BER2, X_IND)
BER4 = tf.reduce_sum(tf.cast(BER3,tf.float32),2)
BER5 = tf.cast(tf.greater(BER4,0),tf.float32)
SER =  tf.reduce_mean(BER5)     

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
    train_step1.run(feed_dict={HY: batch_HY, HH: batch_HH, X: batch_X ,X_IND: x_ind})
    if i % 100== 0 :
        sys.stderr.write(str(i)+' ')
        batch_Y, batch_H, batch_HY, batch_HH, batch_X ,SNR1, H_R, H_I, x_R, x_I, w_R, w_I,x_ind= generate_data_iid_test_no_noise(train_batch_size,K,N,snr_low,snr_high)
        results = sess.run([loss_LS,LOSS[L-1],ber_LS,SER], {HY: batch_HY, HH: batch_HH, X: batch_X,X_IND: x_ind})
        #print('x_ind')
        #print(x_ind)
        #print('S3')
        #print(sess.run(S3[-1], {HY: batch_HY, HH: batch_HH, X: batch_X,X_IND: x_ind}))
        #print('BER4')
        #print(sess.run(BER4, {HY: batch_HY, HH: batch_HH, X: batch_X,X_IND: x_ind}))
        #print('SER')
        #print(sess.run(SER, {HY: batch_HY, HH: batch_HH, X: batch_X,X_IND: x_ind}))
        print_string = [i]+results
        print ' '.join('%s' % x for x in print_string)

for i in range(train_iter): #num of train iter
    batch_Y, batch_H, batch_HY, batch_HH, batch_X , SNR1, H_R, H_I, x_R, x_I, w_R, w_I,x_ind= generate_data_train(train_batch_size,K,N,snr_low,snr_high)
    train_step2.run(feed_dict={HY: batch_HY, HH: batch_HH, X: batch_X,X_IND: x_ind})
    if i % 100 == 0 :
	sys.stderr.write(str(i)+ ' ')
        batch_Y, batch_H, batch_HY, batch_HH, batch_X ,SNR1, H_R, H_I, x_R, x_I, w_R, w_I,x_ind= generate_data_iid_test(train_batch_size,K,N,snr_low,snr_high)
        results = sess.run([loss_LS,LOSS[L-1],ber_LS,BER[L-1]], {HY: batch_HY, HH: batch_HH, X: batch_X,X_IND: x_ind})
        print_string = [i]+results
        print ' '.join('%s' % x for x in print_string)
          
        
        
#start np times
W1 = sess.run(W11)
w1 = sess.run(w11)
W2 = sess.run(W22)
w2 = sess.run(w22)
W3 = sess.run(W33)
w3 = sess.run(w33)
delta_comp = sess.run(delta)


tic = tm.time()

tmp_times_np = np.zeros((test_iter,1)) 

for i_iter in range(test_iter):
    batch_Y, batch_H, batch_HY, batch_HH, batch_X ,SNR1, H_R, H_I, x_R, x_I, w_R, w_I,x_ind= generate_data_iid_test(test_batch_size,K,N,snr_low,snr_high)
    batch_size = np.shape(batch_HY)[0]
    S1_comp=[]
    S1_comp.append(np.zeros([batch_size,2*K]))
    S2_comp=[]
    S2_comp.append(np.zeros([batch_size,16*K]))
    V_comp=[]
    V_comp.append(np.zeros([batch_size,v_size]))
    LOSS_comp=[]
    LOSS_comp.append(np.zeros([]))
    BER_comp=[]
    BER_comp.append(np.zeros([]))
    
    

    
    X_LS_comp = np.matmul(np.expand_dims(batch_HY,1),np.linalg.inv(batch_HH))
    X_LS_comp= np.squeeze(X_LS_comp,1)
    loss_LS_comp = np.mean(np.square(batch_X - X_LS_comp))

    ber_LS_comp = np.mean(np.not_equal(batch_X,np.sign(X_LS_comp)).astype(np.float32))
    tic = tm.time()

    for i in range(0,L-1):

        temp1_comp = np.matmul(np.expand_dims(S1_comp[i], 1), batch_HH)
        temp1_comp = np.squeeze(temp1_comp, 1)

        Z1_comp = S1_comp[-1] -  delta_comp[(i) * 2]*batch_HY + delta_comp[(i) * 2 + 1]*temp1_comp

        Z_comp = np.concatenate((Z1_comp, V_comp[-1]), 1)

        y_temp = np.matmul(Z_comp, W1[i]) + w1[i]
        ZZ_comp = np.maximum(0 , y_temp)

        y_temp = np.matmul(ZZ_comp , W2[i]) + w2[i]

        S2_comp.append(y_temp)

        S2_comp[i+1] = (1-res_alpha)*S2_comp[i+1]+res_alpha*S2_comp[i]
        S2_comp[i+1] = np.clip(S2_comp[i+1], 0, 1)

        y_temp = np.matmul(ZZ_comp, W3[i]) + w3[i]
        
        V_comp.append(y_temp)
        V_comp[i+1] = (1 - res_alpha) * V_comp[i+1] + res_alpha * V_comp[i]

        S3_comp = np.reshape(S2_comp[i+1],[batch_size,K,16])
        
        temp_0_comp = S3_comp[:,:,0]
        temp_1_comp = S3_comp[:,:,1]
        temp_2_comp = S3_comp[:,:,2]
        temp_3_comp = S3_comp[:,:,3]
        temp_4_comp = S3_comp[:,:,4]
        temp_5_comp = S3_comp[:,:,5]
        temp_6_comp = S3_comp[:,:,6]
        temp_7_comp = S3_comp[:,:,7]
        temp_8_comp = S3_comp[:,:,8]
        temp_9_comp = S3_comp[:,:,9]
        temp_10_comp = S3_comp[:,:,10]
        temp_11_comp = S3_comp[:,:,11]
        temp_12_comp = S3_comp[:,:,12]
        temp_13_comp = S3_comp[:,:,13]
        temp_14_comp = S3_comp[:,:,14]    
        temp_15_comp = S3_comp[:,:,15]
    
        S1_real_comp = -3.0*temp_0_comp  +\
              -3.0*temp_1_comp  +\
              -3.0*temp_2_comp  +\
              -3.0*temp_3_comp  +\
              -1.0*temp_4_comp  +\
              -1.0*temp_5_comp  +\
              -1.0*temp_6_comp  +\
              -1.0*temp_7_comp  +\
               1.0*temp_8_comp  +\
               1.0*temp_9_comp  +\
               1.0*temp_10_comp +\
               1.0*temp_11_comp +\
               3.0*temp_12_comp +\
               3.0*temp_13_comp +\
               3.0*temp_14_comp +\
               3.0*temp_15_comp

        S1_im_comp =   -3.0*temp_0_comp  +\
              -1.0*temp_1_comp  +\
               1.0*temp_2_comp  +\
               3.0*temp_3_comp  +\
              -3.0*temp_4_comp  +\
              -1.0*temp_5_comp  +\
               1.0*temp_6_comp  +\
               3.0*temp_7_comp  +\
              -3.0*temp_8_comp  +\
              -1.0*temp_9_comp  +\
               1.0*temp_10_comp +\
               3.0*temp_11_comp +\
              -3.0*temp_12_comp +\
              -1.0*temp_13_comp +\
               1.0*temp_14_comp +\
               3.0*temp_15_comp
               
        S1_comp.append(np.concatenate([S1_real_comp, S1_im_comp], 1))

        X_IND_reshaped_comp = np.reshape(x_ind,[batch_size,16*K])
        LOSS_comp.append(np.log(i)*np.mean(np.mean(np.square(X_IND_reshaped_comp - S2_comp[-1]),1)))
        BER_comp.append(np.mean(np.not_equal(batch_X,np.sign(S1_comp[-1])).astype(np.float32)))


    toc = tm.time()
    tmp_times_np[i_iter] =   (toc-tic)


Max_Val_comp = np.amax(S3_comp,axis=2,keepdims =True)
Greater_comp = np.greater_equal(S3_comp,Max_Val_comp)
BER2_comp = np.round(Greater_comp.astype(np.float32))
x_ind_reshaped = np.reshape(x_ind,[batch_size,K,16])
BER3_comp = np.not_equal(BER2_comp, x_ind_reshaped)
BER4_comp = np.sum(BER3_comp.astype(np.float32),2)
BER5_comp = np.greater(BER4_comp.astype(np.float32),0)
SER_comp =  np.mean(BER5_comp)    

np_time = np.mean(tmp_times_np)/test_batch_size
print('np_time_final')
print(np_time)


print("tf ser at layer is:")
print(np.array(sess.run(SER, {HY: batch_HY, HH: batch_HH, X: batch_X, X_IND:x_ind},))    )
#print(np.array(sess.run(S[1], {HY: batch_HY, HH: batch_HH, X: batch_X}))    )
print("np ser is:")
print(SER_comp)
#print(S_comp[1])



#Testing the trained model
snrdb_list = np.linspace(snrdb_low_test,snrdb_high_test,num_snr)
snr_list = 10.0 ** (snrdb_list/10.0)
bers = np.zeros((4,num_snr))
times = np.zeros((4,num_snr))
tmp_bers = np.zeros((4,test_iter))
tmp_times = np.zeros((4,test_iter))
for j in range(num_snr):
    for jj in range(test_iter):
        print('snr:')
        print(snrdb_list[j])
        print('test iteration:')
        print(jj)
        batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1, H_R, H_I, x_R, x_I, w_R, w_I,x_ind= generate_data_iid_test(test_batch_size , K,N,snr_list[j],snr_list[j])
        tic = tm.time()
        tmp_bers[2,jj] = np.array(sess.run(SER, {HY: batch_HY, HH: batch_HH, X: batch_X,X_IND: x_ind})) 
        toc = tm.time()
        tmp_times[2][jj] =toc - tic
        tmp_bers[0,jj] = np.array(sess.run(ber_LS, {HY: batch_HY, HH: batch_HH, X: batch_X,X_IND: x_ind}))

        tic = tm.time()
        tmp_bers[0,jj] = np.float32(batch_amp13_2(N,K,batch_Y,batch_H,batch_X,n0,test_batch_size,snr_list[j],x_R, x_I))
        toc = tm.time()
        tmp_times[0][jj] =toc - tic
        
        tic = tm.time()
        tmp_bers[3,jj] = np.float32(batch_amp13(N,K,batch_Y,batch_H,batch_X,n0,test_batch_size,snr_list[j],x_R, x_I))
        toc = tm.time()
        tmp_times[3][jj] =toc - tic
        tmp_bers[1,jj] = batch_dfe(batch_Y,batch_H,batch_X,x_R,x_I)
    bers[:,j] = np.mean(tmp_bers,1)
    times[:,j] = np.mean(tmp_times,1)/test_batch_size

print('snrdb_list')
print(snrdb_list)
print('bers')
print(bers)
print('times')
print(times)
print('np_time_final')
print(np_time)

def CreateData(K, N, SNR, B):
    print(SNR)
    H_r = np.random.randn(B, K, N)
    H_i = np.random.randn(B, K, N)
    H_ = np.zeros([B, 2 * K, 2 * N])
    w = np.random.randn(B, 2*N)

    x_R = np.random.randint(4,size = (B,K))
    x_R = x_R * 2
    x_R = x_R - 3
    x_I = np.random.randint(4, size=(B, K))
    x_I = x_I * 2
    x_I = x_I - 3
    x_ = np.concatenate((x_R, x_I), axis=1)
    y_ = np.zeros([B, 2*N])

    for i in range(B):
        # print i
        H = np.concatenate((np.concatenate((H_r[i, :, :], -1 * H_i[i, :, :]), axis=1),
                            np.concatenate((H_i[i, :, :], H_r[i, :, :]), axis=1)), axis=0)
        tmp_snr = (H.T.dot(H)).trace() / (2 * K)
        H = H / np.sqrt(tmp_snr) * np.sqrt(SNR)
        H_[i, :, :] = H
        y_[i, :] = x_[i, :].dot(H)   +w[i,:]

    return y_, H_, x_, SNR, x_R, x_I


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


def sphdec(H, y, symbset, radius):
    global RETVAL
    global TMPVAL
    global SYMBSETSIZE
    global SEARCHFLAG
    global SPHDEC_RADIUS

    Q,R = np.linalg.qr(H,mode='complete')

    z = np.dot(np.transpose(Q),y)

    K = np.shape(H)[1]

    RETVAL = np.zeros((K, 1))
    TMPVAL = np.zeros((K, 1))
    SYMBSETSIZE = len(symbset)
    SEARCHFLAG = 0
    SPHDEC_RADIUS= radius

    sphdec_core(z, R, symbset, K-1, 0)

    if SEARCHFLAG > 0:
        r = RETVAL
    else:
        r = np.ones((K,1))

    return r
print('sphere decoding')
print('16QAM')

test_iter= [10,10,10,10,10,10]
num_snr = 6
snrdb_low_test=8.0
snrdb_high_test=13.0
snrdb_list = np.linspace(snrdb_low_test,snrdb_high_test,num_snr)
snr_list = 10.0 ** (snrdb_list/10.0)
constallation = [-3,-1,1,3]


max_radius=(60*N)/25

for i in range(4):
    sys.stderr.write(str(i) + ' ')
    max_radius = max_radius*1.15
    BERS = np.zeros([num_snr,1])
    SERS = np.zeros([num_snr, 1])
    Times = np.zeros([num_snr,1])
    for j in range(num_snr):
        sys.stderr.write(str(num_snr) + ' ')
        temp_noises = np.zeros([test_iter[j],1])
        temp_ber = 0
        temp_ser = 0
        batch_Y, batch_H, batch_X ,SNR1, x_R, x_I,= CreateData(K, N, snr_list[j], test_iter[j])
        for jj in range(test_iter[j]):
            tic = tm.time()
            xx = sphdec(np.transpose(batch_H[jj]),batch_Y[jj],constallation,max_radius)
            toc = tm.time()
            xx = np.array(xx)
            xx_r = xx[0:K]
            xx_i = xx[K:2*K]

            #print(xx)
            #print(batch_X[jj])
            temp_noises[jj] = toc-tic
            temp_ber = temp_ber +np.sum(np.not_equal(np.transpose(xx),batch_X[jj]))
            temp_ser = temp_ser + np.sum(np.logical_or(np.not_equal(np.transpose(xx_r),x_R[jj]),np.not_equal(np.transpose(xx_i),x_I[jj])))
            #print('aa')
            #print(np.not_equal(np.transpose(xx_r),x_R[jj]))
            #print(np.not_equal(np.transpose(xx_i),x_I[jj]))
            #print(np.logical_or(np.not_equal(np.transpose(xx_r),x_R[jj]),np.not_equal(np.transpose(xx_i),x_I[jj])))
            #print(np.sum(np.logical_or(np.not_equal(np.transpose(xx_r),x_R[jj]),np.not_equal(np.transpose(xx_i),x_I[jj]))))
            #print(temp_ber)
        Times[j] = np.mean(temp_noises)
        temp_ber  = temp_ber/(2.0*K * test_iter[j])
        temp_ser = temp_ser/(1.0*K * test_iter[j])
        BERS[j] = temp_ber
        SERS[j] = temp_ser
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


