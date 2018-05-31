#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import time as tm
import math
import sys
import pickle as pkl

"""
This file is used to train and test the DetNet architecture in the soft decision output scenario.
The constellation used is 8PSK and the channel is complex
all parameters were optimized and trained over the 4X8 iid channel, changing the channel might require parameter tuning

Notice that the run time analysis presented in the paper was made on a numpy version of the tensorflow network.
The times presented in the paper differ from the ones obtained using this code.

writen by Neev Samuel based on the paper:
    "Learning to detect, Neev Samuel,Tzvi Diskin,Ami Wiesel"

contact by neev.samuel@gmail.com

"""

def find_nearest_mpsk_np(value):
    min_dist = 999
    ret_Val = 0
    dist = np.square(value - 1)
    if dist < min_dist:
        min_dist=dist
        ret_Val = 1
    
    dist = np.square(value  - 1.0/np.sqrt(2))
    if dist < min_dist:
        min_dist=dist
        ret_Val = 1.0/np.sqrt(2)
    
    dist = np.square(value - 0)
    if dist < min_dist:
        min_dist=dist
        ret_Val = 0
    
    dist = np.square(value + 1.0/np.sqrt(2))
    if dist < min_dist:
        min_dist=dist
        ret_Val = -1.0/np.sqrt(2)
    
    dist = np.square(value + 1.0)
    if dist < min_dist:
        min_dist=dist
        ret_Val = -1


    return ret_Val

def find_nearest(values):
    values = values + 3
    values = values/2
    values = tf.clip_by_value(values,0,3)
    values = tf.round(values)
    values = values * 2
    values = values - 3
    return values


def validate4(y, SNR, H, K, N):
    final_probs_sqrt2 = np.zeros((K))
    final_probs_one = np.zeros((K))
    final_probs_minus_one = np.zeros((K))
    final_probs_minus_sqrt2 = np.zeros((K))
    final_probs_zero     = np.zeros((K))
    sum_plus_sqrt2  = 0
    sum_plus = 0
    sum_minus = 0
    sum_minus_sqrt2 = 0
    sum_zero = 0
    tmp_snr = (H.T.dot(H)).trace()/ K
    y = y/ np.sqrt(tmp_snr) * np.sqrt(SNR)
    all_pos = np.array(np.meshgrid([1, -1, 0,-1/np.sqrt(2),1/np.sqrt(2)],[1, -1, 0,-1/np.sqrt(2),1/np.sqrt(2)],[1, -1, 0,-1/np.sqrt(2),1/np.sqrt(2)],[1, -1, 0,-1/np.sqrt(2),1/np.sqrt(2)],
                                    [1, -1, 0,-1/np.sqrt(2),1/np.sqrt(2)],[1, -1, 0,-1/np.sqrt(2),1/np.sqrt(2)],[1, -1, 0,-1/np.sqrt(2),1/np.sqrt(2)],[1, -1, 0,-1/np.sqrt(2),1/np.sqrt(2)])).T.reshape(-1,2*4)
    for i in range(len(all_pos)):
        x= all_pos[i]
        tmp_snr = (H.T.dot(H)).trace()/ K
        H_tmp = H / np.sqrt(tmp_snr) * np.sqrt(SNR)
        y_temp = H_tmp.dot(x)

        prob = np.exp(-0.5 * (np.sum(np.power(y_temp[jj] - y[jj], 2) for jj in range(N))))

        for ii in range(K):
            if x[ii] == 1/np.sqrt(2):
                final_probs_sqrt2[ii] = final_probs_sqrt2[ii] + ((1.0 / np.power(2, K)) * prob)
                sum_plus_sqrt2 = sum_plus_sqrt2 + 1
            if x[ii] == 1:
                final_probs_one[ii] = final_probs_one[ii] + ((1.0 / np.power(2, K)) * prob)
                sum_plus = sum_plus + 1
            if x[ii] == -1:
                final_probs_minus_one[ii] = final_probs_minus_one[ii] + ((1.0 / np.power(2, K)) * prob)
                sum_minus = sum_minus + 1
            if x[ii] == -1/np.sqrt(2):
                final_probs_minus_sqrt2[ii] = final_probs_minus_sqrt2[ii] + ((1.0 / np.power(2, K)) * prob)
                sum_minus_sqrt2 = sum_minus_sqrt2 + 1
            if x[ii] == 0:
                final_probs_zero[ii] = final_probs_zero[ii] + ((1.0 / np.power(2, K)) * prob)
                sum_zero = sum_zero + 1                
    for ii in range(K):
        norm = final_probs_one[ii] + final_probs_minus_one[ii] + final_probs_minus_sqrt2[ii] + final_probs_sqrt2[ii] + final_probs_zero[ii]
        final_probs_one[ii] = final_probs_one[ii] / norm
        final_probs_minus_one[ii] = final_probs_minus_one[ii] / norm
        final_probs_minus_sqrt2[ii] = final_probs_minus_sqrt2[ii]/norm
        final_probs_sqrt2[ii] = final_probs_sqrt2[ii]/norm       
        final_probs_zero[ii] = final_probs_zero[ii]/norm

    return final_probs_one, final_probs_minus_one,final_probs_minus_sqrt2,final_probs_sqrt2 , final_probs_zero




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
K = 4
N = 8
snrdb_low = 18.0
snrdb_high = 27.0
snr_low = 10.0 ** (snrdb_low/10.0)
snr_high = 10.0 ** (snrdb_high/10.0)
L = 30
v_size = 4*(2*K)
hl_size = 12*(2*K)
startingLearningRate1 = 0.0005
startingLearningRate2 = 0.0005
decay_factor1 = 0.97
decay_factor2 = 0.97
decay_step_size1 = 1000
decay_step_size2 = 1000
train_iter = 80000
train_iter_no_noise = 10000
n0 = 0.5

train_batch_size = 5000
test_iter= 20
test_batch_size = 40
LOG_LOSS = 1
res_alpha=0.9
num_snr = 8
snrdb_low_test=19.0
snrdb_high_test=26.0
symbols = np.array([1, -1, 0,-1/np.sqrt(2),1/np.sqrt(2)])
normalize_dist = 1

print('8PSK soft with validation')
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
print(train_batch_size)
print(test_iter)
print(test_batch_size)
print(res_alpha)
print(num_snr)
print(snrdb_low_test)
print(snrdb_high_test)
print(normalize_dist)
"""Data generation 

"""
def generate_data(B,K,N,snr_low,snr_high,noise):
    x_bit = np.random.randint(2, size=(B, 3 * K))
    x_r = np.zeros((B, K))
    x_i = np.zeros((B, K))
    x_ind = np.zeros([B,K,8])
    for i in range(B):
        for ii in range(K):
            sym = x_bit[i, 3 * ii:3 * ii + 3]
            if sym[0] == 0 and sym[1] == 0 and sym[2] == 0:
                x_r[i, ii] = -1.0/np.sqrt(2)
                x_i[i, ii] = -1.0/np.sqrt(2)
                x_ind[i,ii,0] = 1
            if sym[0] == 0 and sym[1] == 0 and sym[2] == 1:
                x_r[i, ii] = -1
                x_i[i, ii] = 0
                x_ind[i,ii,1] = 1
            if sym[0] == 0 and sym[1] == 1 and sym[2] == 1:
                x_r[i, ii] = -1.0/np.sqrt(2)
                x_i[i, ii] = 1.0/np.sqrt(2)
                x_ind[i,ii,2] = 1
            if sym[0] == 0 and sym[1] == 1 and sym[2] == 0:
                x_r[i, ii] = 0
                x_i[i, ii] = 1
                x_ind[i,ii,3] = 1
            if sym[0] == 1 and sym[1] == 1 and sym[2] == 0:
                x_r[i, ii] = 1.0/np.sqrt(2)
                x_i[i, ii] = 1.0/np.sqrt(2)
                x_ind[i,ii,4] = 1
            if sym[0] == 1 and sym[1] == 1 and sym[2] == 1:
                x_r[i, ii] = 1
                x_i[i, ii] = 0
                x_ind[i,ii,5] = 1
            if sym[0] == 1 and sym[1] == 0 and sym[2] == 1:
                x_r[i, ii] = 1.0/np.sqrt(2)
                x_i[i, ii] = -1.0/np.sqrt(2)
                x_ind[i,ii,6] = 1
            if sym[0] == 1 and sym[1] == 0 and sym[2] == 0:
                x_r[i, ii] = 0
                x_i[i, ii] = -1
                x_ind[i,ii,7] = 1

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
        H_[i, :, :] = H
        y_[i, :] = H.dot(x_[i, :])   +noise*w[i,:]*np.sqrt(tmp_snr)/np.sqrt(SNR)
        Hy_[i, :] = H.T.dot(y_[i, :])
        HH_[i, :, :] = H.T.dot(H_[i, :, :])
        SNR_[i] = SNR
    return y_,H_,Hy_,HH_,x_,SNR_,x_ind




def piecewise_linear_soft_sign(x):
    t = tf.Variable(0.1)
    y = -1+tf.nn.relu(x+t)/(tf.abs(t)+0.00001)-tf.nn.relu(x-t)/(tf.abs(t)+0.00001)
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
X_IND = tf.placeholder(tf.float32,shape=[None,K,8])
HH = tf.placeholder(tf.float32,shape=[None, 2*K , 2*K])

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
S2.append(tf.zeros([batch_size,8*K]))
S4 = []
S4.append(tf.zeros([batch_size,K,5,2]))
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
    ZZ = relu_layer(Z[-1],(2*K) + v_size , hl_size,'relu'+str(i))

    
    S2_temp = sign_layer(ZZ , hl_size , 8*K,'sign'+str(i))
    
    S2.append(S2_temp)

    
    S2[i]=(1-res_alpha)*S2[i]+res_alpha*S2[i-1]
    S2[i] =    tf.clip_by_value(S2[i],0,1)

    V_temp = affine_layer(ZZ , hl_size , v_size,'aff'+str(i))
    
    V.append(V_temp)

    
    V[i]=(1-res_alpha)*V[i]+res_alpha*V[i-1] 
    
    S3 = tf.reshape(S2[i],[batch_size,K,8])

    temp_0 = S3[:,:,0]
    temp_1 = S3[:,:,1]
    temp_2 = S3[:,:,2]
    temp_3 = S3[:,:,3]
    temp_4 = S3[:,:,4]
    temp_5 = S3[:,:,5]
    temp_6 = S3[:,:,6]
    temp_7 = S3[:,:,7]

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
    
    S4_temp =  tf.zeros([batch_size,K, 5, 2])  

    temp_0 = tf.expand_dims(temp_0,2)
    temp_1 = tf.expand_dims(temp_1,2)
    temp_2 = tf.expand_dims(temp_2,2)
    temp_3 = tf.expand_dims(temp_3,2)
    temp_4 = tf.expand_dims(temp_4,2)
    temp_5 = tf.expand_dims(temp_5,2)
    temp_6 = tf.expand_dims(temp_6,2)
    temp_7 = tf.expand_dims(temp_7,2)

    #                        -1       -1/sqrt(2)          0               1/sqrt(2)         1
    S4_temp1 = tf.concat([temp_1 , temp_0 + temp_2 , temp_3 + temp_7 , temp_4 + temp_6 , temp_5],2)
    S4_temp2 = tf.concat([temp_7 , temp_0 + temp_6 , temp_1 + temp_5 , temp_2 + temp_4 , temp_3],2)
    S4_temp1 = tf.expand_dims(S4_temp1,3)
    S4_temp2 = tf.expand_dims(S4_temp2,3)
    
    S4_temp3 = tf.concat([S4_temp1,S4_temp2],3)
    
    S4.append(S4_temp3)
    X_IND_reshaped = tf.reshape(X_IND,[batch_size,8*K])
    if LOG_LOSS == 1:
        LOSS.append(np.log(i)*tf.reduce_mean(tf.reduce_mean(tf.square(X_IND_reshaped - S2[-1]),1)))
    else:
        LOSS.append(tf.reduce_mean(tf.reduce_mean(tf.square(X_IND_reshaped - S2[-1]),1)))
    BER.append(tf.reduce_mean(tf.cast(tf.not_equal(X_IND, tf.round(S3)), tf.float32)))
Max_Val = tf.reduce_max(S3,axis=2, keep_dims=True)
Greater = tf.greater_equal(S3,Max_Val)
BER2 = tf.round(tf.cast(Greater,tf.float32))
BER3 = tf.not_equal(BER2, X_IND)
BER4 = tf.reduce_sum(tf.cast(BER3,tf.float32),2)
BER5 = tf.cast(tf.greater(BER4,0),tf.float32)
SER =  tf.reduce_mean(BER5)     

TOTAL_LOSS=tf.add_n(LOSS)

saver = tf.train.Saver()

global_step1 = tf.Variable(0, trainable=False)
learning_rate1 = tf.train.exponential_decay(startingLearningRate1, global_step1, decay_step_size1, decay_factor1, staircase=True)
train_step1 = tf.train.AdamOptimizer(learning_rate1).minimize(TOTAL_LOSS)

global_step2 = tf.Variable(0, trainable=False)
learning_rate2 = tf.train.exponential_decay(startingLearningRate2, global_step2, decay_step_size2, decay_factor2, staircase=True)
train_step2 = tf.train.AdamOptimizer(learning_rate2).minimize(TOTAL_LOSS)

init_op=tf.initialize_all_variables()

sess.run(init_op)
for i in range(train_iter_no_noise): #num of train iter
    batch_Y, batch_H, batch_HY, batch_HH, batch_X , SNR1, x_ind= generate_data(train_batch_size,K,N,snr_low,snr_high,0)
    train_step1.run(feed_dict={HY: batch_HY, HH: batch_HH, X: batch_X ,X_IND: x_ind})
    if i % 100== 0 :
        sys.stderr.write(str(i)+' ')
        batch_Y, batch_H, batch_HY, batch_HH, batch_X ,SNR1,x_ind= generate_data(train_batch_size,K,N,snr_low,snr_high,0)
        results = sess.run([loss_LS,LOSS[L-1],ber_LS,SER], {HY: batch_HY, HH: batch_HH, X: batch_X,X_IND: x_ind})
        print_string = [i]+results
        print ' '.join('%s' % x for x in print_string)

for i in range(train_iter): #num of train iter
    batch_Y, batch_H, batch_HY, batch_HH, batch_X , SNR1,x_ind= generate_data(train_batch_size,K,N,snr_low,snr_high,1)
    train_step2.run(feed_dict={HY: batch_HY, HH: batch_HH, X: batch_X,X_IND: x_ind})
    if i % 1000 == 0 :
	sys.stderr.write(str(i)+ ' ')
        batch_Y, batch_H, batch_HY, batch_HH, batch_X ,SNR1,x_ind= generate_data(train_batch_size,K,N,snr_low,snr_high,1)
        results = sess.run([loss_LS,LOSS[L-1],ber_LS,SER], {HY: batch_HY, HH: batch_HH, X: batch_X,X_IND: x_ind})
        print_string = [i]+results
        print ' '.join('%s' % x for x in print_string)
          

#Testing the trained model


snrdb_list = np.linspace(snrdb_low_test,snrdb_high_test,num_snr)
snr_list = 10.0 ** (snrdb_list/10.0)
bers = np.zeros((1,num_snr))
times = np.zeros((1,num_snr))
tmp_bers = np.zeros((1,test_iter))
tmp_times = np.zeros((1,test_iter))
stat_distance = np.zeros((num_snr,2*K))

for j in range(num_snr):
    new_dist_temp = np.zeros((2*K))
    print('snr:')
    print(snrdb_list[j])
    for jj in range(test_iter):
        if jj%100 == 0:
            print('test iteration:')
            print(jj)
        batch_Y, batch_H, batch_HY, batch_HH, batch_X ,SNR1, BATCH_X_IND= generate_data(test_batch_size , K,N,snr_list[j],snr_list[j],1)

        tic = tm.time()
        tmp_bers[0,jj] = np.array(sess.run(SER, {HY: batch_HY, HH: batch_HH, X: batch_X, X_IND:BATCH_X_IND}))
        toc = tm.time()
        tmp_times[0][jj] =toc - tic

        last_layer = np.array(sess.run(S4[-1], {HY: batch_HY, HH: batch_HH, X: batch_X, X_IND:BATCH_X_IND}))
        last_layer = np.clip(last_layer, 0, 1)

        
        for jjj in range(test_batch_size):
            #if K == 3:
            #    final_probs_one, final_probs_minus_one,final_probs_minus_sqrt2,final_probs_sqrt2 , final_probs_zero = validate3(batch_Y[jjj], SNR1[jjj], batch_H[jjj], 2*K, 2*N)
            if K == 4:
                final_probs_one, final_probs_minus_one,final_probs_minus_sqrt2,final_probs_sqrt2 , final_probs_zero = validate4(batch_Y[jjj], SNR1[jjj], batch_H[jjj], 2*K, 2*N)


            NN_lastLayer_minus1_Re = last_layer[jjj,:,0,0]
            NN_lastLayer_minus1_Im = last_layer[jjj,:,0,1]
            
            NN_lastLayer_minus_sqrt2_Re = last_layer[jjj,:,1,0]
            NN_lastLayer_minus_sqrt2_Im = last_layer[jjj,:,1,1]
            
            NN_lastLayer_0_Re = last_layer[jjj,:,2,0]
            NN_lastLayer_0_Im = last_layer[jjj,:,2,1]    

            NN_lastLayer_sqrt2_Re = last_layer[jjj,:,3,0]
            NN_lastLayer_sqrt2_Im = last_layer[jjj,:,3,1]            
            
            NN_lastLayer_1_Re = last_layer[jjj,:,4,0]
            NN_lastLayer_1_Im = last_layer[jjj,:,4,1]

            if normalize_dist:

                Sum_Re = NN_lastLayer_minus1_Re + NN_lastLayer_minus_sqrt2_Re + NN_lastLayer_0_Re + NN_lastLayer_sqrt2_Re + NN_lastLayer_1_Re
                Sum_Im = NN_lastLayer_minus1_Im + NN_lastLayer_minus_sqrt2_Im + NN_lastLayer_0_Im + NN_lastLayer_sqrt2_Im + NN_lastLayer_1_Im
                
                NN_lastLayer_minus1_Re = NN_lastLayer_minus1_Re/Sum_Re
                NN_lastLayer_minus1_Im = NN_lastLayer_minus1_Im/Sum_Im
            
                NN_lastLayer_minus_sqrt2_Re = NN_lastLayer_minus_sqrt2_Re/Sum_Re
                NN_lastLayer_minus_sqrt2_Im = NN_lastLayer_minus_sqrt2_Im/Sum_Im
                
                NN_lastLayer_0_Re = NN_lastLayer_0_Re/Sum_Re
                NN_lastLayer_0_Im = NN_lastLayer_0_Im/Sum_Im

                NN_lastLayer_sqrt2_Re = NN_lastLayer_sqrt2_Re/Sum_Re
                NN_lastLayer_sqrt2_Im = NN_lastLayer_sqrt2_Im/Sum_Im       
            
                NN_lastLayer_1_Re = NN_lastLayer_1_Re/Sum_Re
                NN_lastLayer_1_Im = NN_lastLayer_1_Im/Sum_Im
            
            NN_lastLayer_sqrt2 = np.concatenate([NN_lastLayer_sqrt2_Re, NN_lastLayer_sqrt2_Im],0)
            NN_lastLayer_1 = np.concatenate([NN_lastLayer_1_Re, NN_lastLayer_1_Im],0)
                
            NN_lastLayer_minus_sqrt2 = np.concatenate([NN_lastLayer_minus_sqrt2_Re, NN_lastLayer_minus_sqrt2_Im],0)
            NN_lastLayer_minus1 = np.concatenate([NN_lastLayer_minus1_Re, NN_lastLayer_minus1_Im],0)
            
            NN_lastLayer_zero = np.concatenate([NN_lastLayer_0_Re, NN_lastLayer_0_Im],0)

            new_dist_temp = new_dist_temp + np.absolute(NN_lastLayer_sqrt2 - final_probs_sqrt2) + \
                                            np.absolute(NN_lastLayer_minus_sqrt2 - final_probs_minus_sqrt2) + \
                                            np.absolute(NN_lastLayer_1 - final_probs_one) + \
                                            np.absolute(NN_lastLayer_minus1 - final_probs_minus_one) + \
                                            np.absolute(NN_lastLayer_zero - final_probs_zero)
            
    stat_distance[j] = new_dist_temp/(test_batch_size*test_iter)
    bers[:,j] = np.mean(tmp_bers,1)
    times[:,j] = np.mean(tmp_times[0])/test_batch_size


print('snrdb_list')
print(snrdb_list)
print('bers')
print(bers)
print('times')
print(times)
print('stat_distance')
print(stat_distance)

save_path = saver.save(sess, "./DetNet_Soft_8PSK/8PSK_Soft_model.ckpt")
