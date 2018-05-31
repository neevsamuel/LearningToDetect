#!/usr/bin/env python
"""
This file is used to train and test the DetNet architecture in the hard decision output scenario.
The constellation used is BPSK and the channel is real
all parameters were optimized and trained over the 30X60 iid channel, changing the channel might require parameter tuning

Notice that the run time analysis presented in the paper was made on a numpy version of the tensorflow network.
writen by Neev Samuel based on the paper:
    "Learning to detect, Neev Samuel,Tzvi Diskin,Ami Wiesel"

contact by neev.samuel@gmail.com

"""
import tensorflow as tf
import numpy as np
import time as tm
import math
import sys
import pickle as pkl


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
train_iter =200000
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
        #H=H/np.sqrt(tmp_snr)*np.sqrt(SNR)
        H_[i,:,:]=H
        y_[i,:]=(H.dot(x_[i,:])+w[i,:]*np.sqrt(tmp_snr)/np.sqrt(SNR))
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
        #H=H/np.sqrt(tmp_snr)*np.sqrt(SNR)
        H_[i,:,:]=H
        y_[i,:]=(H.dot(x_[i,:])+w[i,:]*np.sqrt(tmp_snr)/np.sqrt(SNR))
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
    y, W, w = affine_layer(x,input_size,output_size,Layer_num)
    return y,W,w

#tensorflow placeholders, the input given to the model in order to train and test the network
HY = tf.placeholder(tf.float32,shape=[None,K])
X = tf.placeholder(tf.float32,shape=[None,K])
HH = tf.placeholder(tf.float32,shape=[None, K , K])
X_IND = tf.placeholder(tf.float32,shape=[None, K , 2])

batch_size = tf.shape(HY)[0]

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
BER3 = tf.not_equal(BER2, X_IND)
BER4 = tf.reduce_sum(tf.cast(BER3,tf.float32),2)
BER5 = tf.cast(tf.greater(BER4,0),tf.float32)
SER =  tf.reduce_mean(BER5)    
TOTAL_LOSS=tf.add_n(LOSS)

saver = tf.train.Saver()


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
        results = sess.run([LOSS[L-1],SER], {HY: batch_HY, HH: batch_HH, X: batch_X, X_IND:x_ind})
        print_string = [i]+results
        print ' '.join('%s' % x for x in print_string)


batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1,x_ind = generate_data_iid_test(test_batch_size, K, N, 10,10)

batch_size = np.shape(batch_HY)[0]
X_LS_comp = np.matmul(np.expand_dims(batch_HY,1),np.linalg.inv(batch_HH))
X_LS_comp= np.squeeze(X_LS_comp,1)
loss_LS_comp = np.mean(np.square(batch_X - X_LS_comp))

ber_LS_comp = np.mean(np.not_equal(batch_X,np.sign(X_LS_comp)).astype(np.float32))


#saver.restore(sess, "./DetNet_HD_BPSK/BPSK_HD_model.ckpt")

#Testing the trained model
snrdb_list = np.linspace(snrdb_low_test,snrdb_high_test,num_snr)
snr_list = 10.0 ** (snrdb_list/10.0)
bers = np.zeros((1,num_snr))
times = np.zeros((1,num_snr))
tmp_bers = np.zeros((1,test_iter))
tmp_times = np.zeros((1,test_iter))
tmp_ber_iter=np.zeros([L,test_iter])
ber_iter=np.zeros([L,num_snr])
for j in range(num_snr):
    for jj in range(test_iter):
        print('snr:')
        print(snrdb_list[j])
        print('test iteration:')
        print(jj)
        batch_Y, batch_H, batch_HY, batch_HH, batch_X ,SNR1,x_ind= generate_data_iid_test(test_batch_size , K,N,snr_list[j],snr_list[j])
        results = sess.run([ LOSS[L - 1],  BER[L - 1]], {HY: batch_HY, HH: batch_HH, X: batch_X, X_IND:x_ind})

        tic = tm.time()
        tmp_ber_iter[:, jj] = np.array(sess.run(SER, {HY: batch_HY, HH: batch_HH, X: batch_X, X_IND:x_ind}))
        toc = tm.time()
        tmp_times[0][jj] = toc - tic
        tmp_bers[0][jj] = results[1]

    bers[0][j] = np.mean(tmp_bers[0])
    times[0][j] = np.mean(tmp_times[0]) / test_batch_size
    ber_iter[:, j] = np.mean(tmp_ber_iter, 1)

print('snrdb_list')
print(snrdb_list)
print('bers')
print(bers)
print('times')
print(times)

save_path = saver.save(sess, "./DetNet_HD_BPSK/BPSK_HD_model.ckpt")
