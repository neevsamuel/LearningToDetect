#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import time as tm
import math
import sys
import pickle as pkl


def sdr_ip(y,H):
    epsilon = 1e-5
    K=np.shape(H)[1]
    Q=np.array(np.bmat([[np.dot(H.T,H),-np.dot(H.T,y)],[-np.dot(H.T,y).T,np.dot(y.T,y)]]))
    Q = -Q;
    n = np.shape(Q)[1]
    X = np.identity(n)
    lambd = 1.1*np.dot(np.absolute(Q),np.ones((n,1)))
    Z = np.diag(np.reshape(lambd,n)) - Q

    k = 0

    while(np.trace(np.dot(Z,X)) > epsilon):
        mu = np.trace(np.dot(Z,X))/n
        mu = mu/2

        #compute newton search direction
        W = np.linalg.inv(Z)
        T = np.multiply(W,X)

        dlambd = np.linalg.solve(T, np.reshape(mu*np.diag(W) - np.ones(n) ,(n,1)))
        dZ = np.diag(np.reshape(dlambd,n,1))
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
        x=np.sign(X[range(K),-1]).T
    return (x)

def ampF1(s,tau):
    return np.tanh(s/tau)

def ampG1(s,tau):
    return 1-(np.tanh(s/tau) ** 2)

def amp(y,H,N0,K,N):

    L = K*3
    beta = K/(0.+N)
    s = np.zeros(K)
    tau = beta*1/N0
    r=y
    for it in range(L):
        z = s+np.dot(H.T,r)
        s = ampF1(z,N0*(1.0+tau))
        tau_new = beta/N0*np.mean(ampG1(z,N0*(1.0+tau)))
        r = y - np.dot(H,s)+tau_new/(1.0+tau)*r
        tau = tau_new
    return np.sign(s)

def batch_amp(batch_Y,batch_H,batch_X,n0,B,SNR,K,N):
    err_amp = 0.0
    for i in range(B):
	xx = amp(batch_Y[i]/np.sqrt(SNR[i]),batch_H[i]/np.sqrt(SNR[i]),n0,K,N)
	#print 'xx'
	#print xx
	#print 'x'
	#print batch_X[i]
	#print err_amp
	err_amp+=(np.mean(np.not_equal(batch_X[i],xx)))/B
	err_amp
    return err_amp

def dfe(y,H):
    Q,R = np.linalg.qr(H)
    Qy = np.dot (Q.T,y)
    xx=np.zeros([K])
    for k in range(K-1,-1,-1):
        xx[k]=np.sign((Qy[k]-np.dot(R[k][k:],xx[k:]))/R[k][k])
    return(xx)

def batch_dfe(y,H,x,N,K):
    B = np.shape(y)[0]
    ber=0
    for i in range(B):
        xx = dfe(y[i].T,H[i])
        ber+=np.mean(np.not_equal(x[i],xx))
    ber=ber/B
    return np.float32(ber)


def batch_sdr(y,H,x,N,K):
    B = np.shape(y)[0]
    ber=0.0
    for i in range(B):
        xx = sdr_ip(np.reshape(y[i],(len(y[i]),1)),H[i])
        ber+=np.mean(np.not_equal(x[i],xx))
    ber=ber/B
    return np.float32(ber)
###start here
"""
Parameters
K - size of x
N - size of y
snrdb_low - the lower bound of noise db used during training
snr_high - the higher bound of noise db used during training
L - number of layers in DetNet
v_size = size of auxiliary variable at each layer
hl_size - size of hidden layer at each DetNet layer (the dimention the layers input are increased to
startingLearningRate - the initial step size of the gradient descent algorithm
decay_factor & decay_step_size - each decay_step_size steps the learning rate decay by decay_factor
train_iter - number of train iterations
train_batch_size - batch size during training phase
test_iter - number of test iterations
test_batch_size  - batch size during testing phase
LOG_LOSS - equal 1 if loss of each layer should be sumed in proportion to the layer depth, otherwise all losses have the same weight 
res_alpha- the proportion of the previuos layer output to be added to the current layers output (view ResNet article)
snrdb_low_test & snrdb_high_test & num_snr - when testing, num_snr different SNR values will be tested, uniformly spread between snrdb_low_test and snrdb_high_test 
"""
sess = tf.InteractiveSession()

#parameters
K = 30
N = 60
snrdb_low = 7.0
snrdb_high = 14.0
snr_low = 10.0 ** (snrdb_low/10.0)
snr_high = 10.0 ** (snrdb_high/10.0)
L=30
v_size = 2*K
hl_size = 4*K
startingLearningRate = 0.0008
decay_factor = 0.97
decay_step_size = 1000
train_iter =1
train_batch_size = 3000
test_iter= 200
test_batch_size = 2000
LOG_LOSS = 1
res_alpha=0.9
num_snr = 6
snrdb_low_test=8.0
snrdb_high_test=13.0

print('BPSK one hot + times no norm')
print(K)
print(N)
print(snrdb_low)
print(snrdb_high)
print(snr_low)
print(snr_high)
print(L)
print(v_size)
print(hl_size)
print(startingLearningRate)
print(decay_factor)
print(decay_step_size)
print(train_iter)
print(train_batch_size)
print(test_iter)
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
def generate_data_iid_test(B,K,N,snr_low,snr_high):
    H_=np.random.randn(B,N,K)
    x_=np.sign(np.random.rand(B,K)-0.5)
    y_=np.zeros([B,N])
    w=np.random.randn(B,N)
    Hy_=x_*0
    HH_=np.zeros([B,K,K])
    SNR_= np.zeros([B])
    x_ind = np.zeros([B,K,2])
    for i in range(B):
        for ii in range(K):
            if x_[i][ii] == 1:
                x_ind[i][ii][0] = 1
            if x_[i][ii] == -1:
                x_ind[i][ii][1] = 1         
    for i in range(B):
        SNR = np.random.uniform(low=snr_low,high=snr_high)
        H=H_[i,:,:]
        tmp_snr=(H.T.dot(H)).trace()/K
        H=H/np.sqrt(tmp_snr)*np.sqrt(SNR)
        H_[i,:,:]=H
        y_[i,:]=(H.dot(x_[i,:])+w[i,:])#*np.sqrt(tmp_snr)/np.sqrt(SNR))
        Hy_[i,:]=H.T.dot(y_[i,:])
        HH_[i,:,:]=H.T.dot( H_[i,:,:])
        SNR_[i] = SNR
    return y_,H_,Hy_,HH_,x_,SNR_,x_ind

def generate_data_train(B,K,N,snr_low,snr_high):
    H_=np.random.randn(B,N,K)
    x_=np.sign(np.random.rand(B,K)-0.5)
    y_=np.zeros([B,N])
    w=np.random.randn(B,N)
    Hy_=x_*0
    HH_=np.zeros([B,K,K])
    SNR_= np.zeros([B])
    x_ind = np.zeros([B,K,2])
    for i in range(B):
        for ii in range(K):
            if x_[i][ii] == 1:
                x_ind[i][ii][0] = 1
            if x_[i][ii] == -1:
                x_ind[i][ii][1] = 1   
    for i in range(B):
        SNR = np.random.uniform(low=snr_low,high=snr_high)
        H=H_[i,:,:]
        tmp_snr=(H.T.dot(H)).trace()/K
        H=H/np.sqrt(tmp_snr)*np.sqrt(SNR)
        H_[i,:,:]=H
        y_[i,:]=(H.dot(x_[i,:])+w[i,:])#*np.sqrt(tmp_snr)/np.sqrt(SNR))
        Hy_[i,:]=H.T.dot(y_[i,:])
        HH_[i,:,:]=H.T.dot( H_[i,:,:])
        SNR_[i] = SNR
    return y_,H_,Hy_,HH_,x_,SNR_,x_ind


def piecewise_linear_soft_sign(x):
    t = tf.Variable(0.1)
    y = -1+tf.nn.relu(x+t)/(tf.abs(t)+0.00001)-tf.nn.relu(x-t)/(tf.abs(t)+0.00001)
    return y

def affine_layer(x,input_size,output_size,Layer_num):
    W = tf.Variable(tf.random_normal([input_size, output_size], stddev=0.01),name = 'W'+Layer_num)
    w = tf.Variable(tf.random_normal([1, output_size], stddev=0.01),name = 'w'+Layer_num)
    y = tf.matmul(x, W)+w
    return y,W,w

def relu_layer(x,input_size,output_size,Layer_num):
    y,W,w = affine_layer(x,input_size,output_size,Layer_num)
    y = tf.nn.relu(y)

    return y,W,w

def sign_layer(x,input_size,output_size,Layer_num):
    #y = piecewise_linear_soft_sign(affine_layer(x,input_size,output_size,Layer_num))
    y, W, w = affine_layer(x,input_size,output_size,Layer_num)
    return y,W,w

#tensorflow placeholders, the input given to the model in order to train and test the network
HY = tf.placeholder(tf.float32,shape=[None,K])
X = tf.placeholder(tf.float32,shape=[None,K])
HH = tf.placeholder(tf.float32,shape=[None, K , K])
X_IND = tf.placeholder(tf.float32,shape=[None, K , 2])

batch_size = tf.shape(HY)[0]
X_LS = tf.matmul(tf.expand_dims(HY,1),tf.matrix_inverse(HH))
X_LS= tf.squeeze(X_LS,1)
loss_LS = tf.reduce_mean(tf.square(X - X_LS))
ber_LS = tf.reduce_mean(tf.cast(tf.not_equal(X,tf.sign(X_LS)), tf.float32))


S2=[]
S2.append(tf.zeros([batch_size,2*K]))
S1=[]
S1.append(tf.zeros([batch_size,K]))
S_NO_res=[]
S_NO_res.append(tf.zeros([batch_size,K]))
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
ZZ_test = []
#The architecture of DetNet
for i in range(1,L):
    temp1 = tf.matmul(tf.expand_dims(S1[-1],1),HH)
    temp1= tf.squeeze(temp1,1)
    Z1 = S1[-1] - delta[(i-1) * 2]*HY + delta[(i-1) * 2 + 1]*temp1
    Z = tf.concat([Z1, V[-1]], 1)
    ZZ,Wtemp,wtemp = relu_layer(Z,K + v_size , hl_size,'relu'+str(i))

    W11.append(Wtemp)
    w11.append(wtemp)
    ZZ1.append((ZZ))

    S2_temp,W22_temp,w22_temp = sign_layer(ZZ, hl_size, 2*K, 'sign' + str(i))
    S2.append(S2_temp)
    W22.append(W22_temp)
    w22.append(w22_temp)

    S_NO_res.append(S2[i])
    S2[i]=(1-res_alpha)*S2[i]+res_alpha*S2[i-1]
    #S2[i] = tf.clip_by_value(S2[i],0,1)
    V_temp ,  W33_temp , w33_temp=affine_layer(ZZ , hl_size , v_size,'aff'+str(i))
    V.append(V_temp)
    W33.append(W33_temp)
    w33.append(w33_temp)

    V[i]=(1-res_alpha)*V[i]+res_alpha*V[i-1]  
    
    S3 = tf.reshape(S2[i],[batch_size,K,2])
    
    temp_0 = S3[:,:,0]
    temp_1 = S3[:,:,1]
    
    temp_2 = 1*temp_0 + (-1)*temp_1
    S1.append(temp_2)
    X_IND_reshaped = tf.reshape(X_IND,[batch_size,2*K])
    if LOG_LOSS == 1:
        LOSS.append(np.log(i)*tf.reduce_mean(tf.reduce_mean(tf.square(X_IND_reshaped - S2[-1]),1)))
    else:
        LOSS.append(tf.reduce_mean(tf.reduce_mean(tf.square(X_IND_reshaped - S2[-1]),1)))      
        
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


global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(startingLearningRate, global_step, decay_step_size, decay_factor, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(TOTAL_LOSS)
init_op=tf.initialize_all_variables()

sess.run(init_op)
#Training DetNet
for i in range(train_iter): #num of train iter
    batch_Y, batch_H, batch_HY, batch_HH, batch_X , SNR1,x_ind= generate_data_train(train_batch_size,K,N,snr_low,snr_high)
    train_step.run(feed_dict={HY: batch_HY, HH: batch_HH, X: batch_X, X_IND:x_ind})
    if i % 1000 == 0 :
        sys.stderr.write(str(i)+' ')
        batch_Y, batch_H, batch_HY, batch_HH, batch_X ,SNR1,x_ind= generate_data_iid_test(train_batch_size,K,N,snr_low,snr_high)
        results = sess.run([loss_LS,LOSS[L-1],ber_LS,SER], {HY: batch_HY, HH: batch_HH, X: batch_X, X_IND:x_ind})
        print_string = [i]+results
        print ' '.join('%s' % x for x in print_string)


batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1,x_ind = generate_data_iid_test(test_batch_size, K, N, 10,10)

batch_size = np.shape(batch_HY)[0]
X_LS_comp = np.matmul(np.expand_dims(batch_HY,1),np.linalg.inv(batch_HH))
X_LS_comp= np.squeeze(X_LS_comp,1)
loss_LS_comp = np.mean(np.square(batch_X - X_LS_comp))
#print('ls comp')
#print(np.sign(X_LS_comp))
#print('batch_X')
#print(batch_X)
#print("not equal")
#print(np.not_equal(batch_X,np.sign(X_LS_comp)))

#print(np.not_equal(batch_X,np.sign(X_LS_comp)).astype(np.float32))
ber_LS_comp = np.mean(np.not_equal(batch_X,np.sign(X_LS_comp)).astype(np.float32))
#print(ber_LS_comp)

print("tf ber at ls is:")
print(np.array(sess.run(ber_LS, {HY: batch_HY, HH: batch_HH, X: batch_X, X_IND:x_ind}))    )
print("np ber at ls is:")
print(ber_LS_comp)

print("tf loss at ls is:")
print(np.array(sess.run(loss_LS, {HY: batch_HY, HH: batch_HH, X: batch_X, X_IND:x_ind}))    )
print("np loss at ls is:")
print(loss_LS_comp)

W1 = sess.run(W11)
w1 = sess.run(w11)
W2 = sess.run(W22)
w2 = sess.run(w22)
W3 = sess.run(W33)
w3 = sess.run(w33)


S1_comp=[]
S1_comp.append(np.zeros([batch_size,K]))
S2_comp=[]
S2_comp.append(np.zeros([batch_size,2*K]))
V_comp=[]
V_comp.append(np.zeros([batch_size,v_size]))
LOSS_comp=[]
LOSS_comp.append(np.zeros([]))
BER_comp=[]
BER_comp.append(np.zeros([]))
tic = tm.time()
#for i in range(0,L-1):
#    print("layer")
#    print(i)
#    temp1_comp = np.matmul(np.expand_dims(S1_comp[i], 1), batch_HH)
#    temp1_comp = np.squeeze(temp1_comp, 1)
#
#
#
#
#    Z_comp = np.concatenate((batch_HY, S1_comp[-1], temp1_comp, V_comp[-1]), 1)
#
#
#    y_temp = np.matmul(Z_comp, W1[i]) + w1[i]
#    ZZ_comp = np.maximum(0 , y_temp)
#
#    y_temp = np.matmul(ZZ_comp , W2[i]) + w2[i]
#
#    S2_comp.append(y_temp)
#
#    S2_comp[i+1]=(1-res_alpha)*S2_comp[i+1]+res_alpha*S2_comp[i]
#    S2_comp[i+1] =  np.clip(S2_comp[i+1], 0, 1)
#
#    y_temp = np.matmul(ZZ_comp, W3[i]) + w3[i]
#
#    V_comp.append(y_temp)
#    V_comp[i+1] = (1 - res_alpha) * V_comp[i+1] + res_alpha * V_comp[i]
#
#    S3_comp = np.reshape(S2_comp[i],[batch_size,K,2])
#    
#
#    temp_0_comp = S3_comp[:,:,0]
#    temp_1_comp = S3_comp[:,:,1]
#    
#    temp_2_comp = 1*temp_0_comp + (-1)*temp_1_comp
#    S1_comp.append(temp_2_comp)
#    X_IND_reshaped_comp = np.reshape(x_ind,[batch_size,2*K])
#    LOSS_comp.append(np.log(i)*np.mean(np.mean(np.square(X_IND_reshaped_comp - S2_comp[-1]),1)))
#    BER_comp.append(np.mean(np.not_equal(batch_X,np.sign(S1_comp[-1])).astype(np.float32)))
#
#
#BER2_comp = np.round(S3_comp)
#BER3_comp = np.not_equal(BER2_comp, x_ind)
#BER4_comp = np.sum(BER3_comp.astype(np.float32),2)
#BER5_comp = np.greater(BER4_comp.astype(np.float32),0)
#SER_comp =  np.mean(BER5_comp)    
#
toc = tm.time()
time_np = (toc-tic)/test_batch_size
print('time np')
print(time_np)



print("tf ser at layer is:")
print(np.array(sess.run(SER, {HY: batch_HY, HH: batch_HH, X: batch_X, X_IND:x_ind},))    )
#print(np.array(sess.run(S[1], {HY: batch_HY, HH: batch_HH, X: batch_X}))    )
#print("np ser is:")
#print(SER_comp)
#print(S_comp[1])

#Testing the trained model
snrdb_list = np.linspace(snrdb_low_test,snrdb_high_test,num_snr)
snr_list = 10.0 ** (snrdb_list/10.0)
bers = np.zeros((5,num_snr))
times = np.zeros((5,num_snr))
tmp_bers = np.zeros((5,test_iter))
tmp_times = np.zeros((5,test_iter))
tmp_ber_iter=np.zeros([L,test_iter])
ber_iter=np.zeros([L,num_snr])
for j in range(num_snr):
    for jj in range(test_iter):
        print('snr:')
        print(snrdb_list[j])
        print('test iteration:')
        print(jj)
        batch_Y, batch_H, batch_HY, batch_HH, batch_X ,SNR1,x_ind= generate_data_iid_test(test_batch_size , K,N,snr_list[j],snr_list[j])
        results = sess.run([loss_LS, LOSS[L - 1], ber_LS, BER[L - 1]], {HY: batch_HY, HH: batch_HH, X: batch_X, X_IND:x_ind})
        # ber_LS = tf.cast(np.mean(np.logical_or(np.not_equal(x_real, xLS_real), np.not_equal(x_imag, xLS_imag))),
        #                 tf.float32)
        tic = tm.time()
        tmp_ber_iter[:, jj] = np.array(sess.run(SER, {HY: batch_HY, HH: batch_HH, X: batch_X, X_IND:x_ind}))
        toc = tm.time()
        tmp_times[2][jj] = toc - tic
        results.append(batch_dfe(batch_Y, batch_H, batch_X, N, K))
        

        tic = tm.time()
        tmp_bers[1][jj] = batch_dfe(batch_Y, batch_H, batch_X, N, K)
        toc = tm.time()
        tmp_times[1][jj] = toc - tic

        tmp_bers[0][jj] = results[2]
        tmp_bers[2][jj] = results[3]
        tic = tm.time()
        tmp_bers[3][jj]=batch_sdr(batch_Y,batch_H,batch_X,N,K)
        toc = tm.time()
        tmp_times[3][jj] = toc - tic

        tic = tm.time()
        n0 = 0.27

        tmp_bers[4][jj] = batch_amp(batch_Y, batch_H, batch_X, n0, test_batch_size, SNR1, K, N)
        toc = tm.time()
        tmp_times[4][jj] = toc - tic

    bers[0][j] = np.mean(tmp_bers[0])
    bers[1][j] = np.mean(tmp_bers[1])
    bers[2][j] = np.mean(tmp_bers[2])
    bers[3][j] = np.mean(tmp_bers[3])
    bers[4][j] = np.mean(tmp_bers[4])
    times[1][j] = np.mean(tmp_times[1]) / test_batch_size
    times[2][j] = np.mean(tmp_times[2]) / test_batch_size
    times[3][j] = np.mean(tmp_times[3]) / test_batch_size
    times[4][j] = np.mean(tmp_times[4]) / test_batch_size
    ber_iter[:, j] = np.mean(tmp_ber_iter, 1)

print('snrdb_list')
print(snrdb_list)
print('bers')
print(bers)
print('times')
print(times)
print('time np')
print(time_np)

import numpy as np
from copy import deepcopy
import time as tm

def CreateData(K, N, SNR, B):
    print(SNR)
    H_ = np.random.randn(B, K, N)
    w = np.random.randn(B, N)
    x_ = np.sign(np.random.rand(B, K) - 0.5)
    y_ = np.zeros([B, N])
    SNR_= np.zeros([B])

    for i in range(B):
        SNR = np.random.uniform(low=SNR, high=SNR)
        H = H_[i, :, :]
        tmp_snr = (H.T.dot(H)).trace() / (1.0*K)
        H = H / np.sqrt(tmp_snr) * np.sqrt(SNR)
        H_[i,:, :] = H
        y_[i, :] = x_[i, :].dot(H)+ w[i, :]
        SNR_[i] = SNR
    return y_, H_, x_, SNR_


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
        r = 0

    return r

print('Sphere Decoding')

test_iter= [3000,5000,8000,10000,15000,20000]
num_snr = 6
snrdb_low_test=8.0
snrdb_high_test=13.0
snrdb_list = np.linspace(snrdb_low_test,snrdb_high_test,num_snr)
snr_list = 10.0 ** (snrdb_list/10.0)
constallation = [-1,1]


max_radius=40

for i in range(6):
    max_radius = max_radius*1.2
    BERS = np.zeros([num_snr,1])
    Times = np.zeros([num_snr,1])
    for j in range(num_snr):
        temp_noises = np.zeros([test_iter[j],1])
        temp_ber = 0
        batch_Y, batch_H, batch_X ,SNR1= CreateData(K, N, snr_list[j], test_iter[j])
        for jj in range(test_iter[j]):
            tic = tm.time()
            xx = sphdec(np.transpose(batch_H[jj]),batch_Y[jj],constallation,max_radius)
            toc = tm.time()
            #print(xx)
            #print(batch_X[jj])
            temp_noises[jj] = toc-tic
            temp_ber = temp_ber +np.sum(np.not_equal(np.transpose(xx),batch_X[jj]))

            #print(temp_ber)
        Times[j] = np.mean(temp_noises)
        temp_ber  = temp_ber/(1.0*K * test_iter[j])
        BERS[j] = temp_ber
    print('i')
    print(i)
    print('max_radius')
    print(max_radius)
    print('BERS')
    print(BERS)
    print('Times')
    print(Times)


