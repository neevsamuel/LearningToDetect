#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import time as tm
import math
import sys
import pickle as pkl


"""
This file is used to train and test the DetNet architecture in the hard decision output scenario.
The constellation used is 8PSK and the channel is complex
all parameters were optimized and trained over the 15X25 iid channel, changing the channel might require parameter tuning

Notice that the run time analysis presented in the paper was made on a numpy version of the tensorflow network.
writen by Neev Samuel based on the paper:
    "Learning to detect, Neev Samuel,Tzvi Diskin,Ami Wiesel"

contact by neev.samuel@gmail.com

"""
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



###start here
sess = tf.InteractiveSession()
#parameters
"""
Parameters
K - size of x
N - size of y
snrdb_low - the lower bound of noise db used during training
snr_high - the higher bound of noise db used during training
L - number of layers in DetNet
v_size = size of auxiliary variable at each layer
hl_size - size of hidden layer at each DetNet layer (the dimention the layers input are increased to
startingLearningRate1 - the initial step size of the gradient descent algorithm when train phase without noise
startingLearningRate2 - the initial step size of the gradient descent algorithm when train phase with noise
decay_factor & decay_step_size - each decay_step_size steps the learning rate decay by decay_factor
train_iter - number of train iterations
train_iter_no_noise - number of train iterations without noise
train_batch_size - batch size during training phase
test_iter - number of test iterations
test_batch_size  - batch size during testing phase
LOG_LOSS - equal 1 if loss of each layer should be sumed in proportion to the layer depth, otherwise all losses have the same weight 
res_alpha- the proportion of the previuos layer output to be added to the current layers output (view ResNet article)
snrdb_low_test & snrdb_high_test & num_snr - when testing, num_snr different SNR values will be tested, uniformly spread between snrdb_low_test and snrdb_high_test 
symbols - the possible symbols the consellation has (when converted to the real setting as discribed in the paper)
"""
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
train_iter = 200000
train_iter_no_noise = 5000
n0 = 0.5

train_batch_size = 2000
test_iter= 100
test_batch_size = 2000
LOG_LOSS = 1
res_alpha=0.9
num_snr = 6
snrdb_low_test=19.0
snrdb_high_test=24.0

print('8PSK DetNet parameters:')
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


def generate_data_iid_test(B,K,N,snr_low,snr_high,WithNoise):
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
        SNR = np.random.uniform(low=snr_low, high=snr_high)
        H = np.concatenate((np.concatenate((H_R[i, :, :], -1 * H_I[i, :, :]), axis=1),
                            np.concatenate((H_I[i, :, :], H_R[i, :, :]), axis=1)), axis=0)
        tmp_snr = (H.T.dot(H)).trace() / (2 * K)
        H_[i, :, :] = H
        y_[i, :] = H.dot(x_[i, :]) + WithNoise*w[i,:]*np.sqrt(tmp_snr)/np.sqrt(SNR)
        Hy_[i, :] = H.T.dot(y_[i, :])
        HH_[i, :, :] = H.T.dot(H_[i, :, :])
        SNR_[i] = SNR
    return y_,H_,Hy_,HH_,x_,SNR_, H_R, H_I, x_r, x_i, w_R, w_I,x_ind

def generate_data_train(B,K,N,snr_low,snr_high,WithNoise):
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
        SNR = np.random.uniform(low=snr_low, high=snr_high)
        H = np.concatenate((np.concatenate((H_R[i, :, :], -1 * H_I[i, :, :]), axis=1),
                            np.concatenate((H_I[i, :, :], H_R[i, :, :]), axis=1)), axis=0)
        tmp_snr = (H.T.dot(H)).trace() / (2 * K)
        H_[i, :, :] = H
        y_[i, :] = H.dot(x_[i, :])   +WithNoise*w[i,:]*np.sqrt(tmp_snr)/np.sqrt(SNR)
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
    #t = tf.Variable(0.1)
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
    y = affine_layer(x,input_size,output_size,Layer_num)
    return y

#tensorflow placeholders, the input given to the model in order to train and test the network
HY = tf.placeholder(tf.float32,shape=[None,2*K])
X = tf.placeholder(tf.float32,shape=[None,2*K])
HH = tf.placeholder(tf.float32,shape=[None, 2*K , 2*K])
X_IND = tf.placeholder(tf.float32,shape=[None,8*K])


batch_size = tf.shape(HY)[0]


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

"""
The architecture of DetNet
"""
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
        LOSS.append(np.log(i) * tf.reduce_mean(tf.reduce_mean(tf.square(X_IND - S2[-1]), 1) ))
    else:
        LOSS.append(tf.reduce_mean(tf.reduce_mean(tf.square(X_IND - S2[-1]), 1)))
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

saver = tf.train.Saver()

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
    batch_Y, batch_H, batch_HY, batch_HH, batch_X , SNR1, H_R, H_I, x_R, x_I, w_R, w_I,x_ind= generate_data_train(train_batch_size,K,N,snr_low,snr_high,0)
    train_step1.run(feed_dict={HY: batch_HY, HH: batch_HH, X: batch_X,X_IND:x_ind})
    if i % 1000 == 0 :
        sys.stderr.write(str(i)+' ')
        batch_Y, batch_H, batch_HY, batch_HH, batch_X ,SNR1, H_R, H_I, x_R, x_I, w_R, w_I,x_ind= generate_data_iid_test(train_batch_size,K,N,snr_low,snr_high,0)
        results = sess.run([LOSS[L-1],SER], {HY: batch_HY, HH: batch_HH, X: batch_X,X_IND:x_ind})
        print_string = [i]+results
        print ' '.join('%s' % x for x in print_string)


for i in range(train_iter): #num of train iter
    batch_Y, batch_H, batch_HY, batch_HH, batch_X , SNR1, H_R, H_I, x_R, x_I, w_R, w_I,x_ind= generate_data_train(train_batch_size,K,N,snr_low,snr_high,1)
    train_step2.run(feed_dict={HY: batch_HY, HH: batch_HH, X: batch_X,X_IND:x_ind})
    if i % 1000 == 0 :
	sys.stderr.write(str(i)+ ' ')
        batch_Y, batch_H, batch_HY, batch_HH, batch_X ,SNR1, H_R, H_I, x_R, x_I, w_R, w_I,x_ind= generate_data_iid_test(train_batch_size,K,N,snr_low,snr_high,1)
        results = sess.run([LOSS[L-1],SER], {HY: batch_HY, HH: batch_HH, X: batch_X,X_IND:x_ind})
        print_string = [i]+results
        print ' '.join('%s' % x for x in print_string)

#saver.restore(sess, "./DetNet_HD_8PSK/8PSK_HD_model.ckpt")

#Testing the trained model
snrdb_list = np.linspace(snrdb_low_test,snrdb_high_test,num_snr)
snr_list = 10.0 ** (snrdb_list/10.0)
bers = np.zeros((1,num_snr))
sers = np.zeros((1,num_snr))
times = np.zeros((1,num_snr))
tmp_bers = np.zeros((1,test_iter))
tmp_sers = np.zeros((1,test_iter))

tmp_times = np.zeros((4,test_iter))
for j in range(num_snr):
    for jj in range(test_iter):
        sys.stderr.write(str(jj) + ' ')
        print('snr:')
        print(snrdb_list[j])
        print('test iteration:')
        print(jj)

        batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1, H_R, H_I, x_R, x_I, w_R, w_I,x_ind= generate_data_iid_test(test_batch_size , K,N,snr_list[j],snr_list[j],1)
        tic = tm.time()
        tmp_bers[0,jj] = np.array(sess.run(SER, {HY: batch_HY, HH: batch_HH, X: batch_X,X_IND:x_ind}))/(test_batch_size*K)
        toc = tm.time()
        tmp_times[0][jj] =toc - tic


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

save_path = saver.save(sess, "./DetNet_HD_8PSK/8PSK_HD_model.ckpt")

