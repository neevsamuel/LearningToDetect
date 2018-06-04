#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import time as tm
import math
import sys
import pickle as pkl

def validate(y,SNR,H,K,N,Real_X):
    final_probs_one = np.zeros((K))
    final_probs_minus_one = np.zeros((K))
    sum_plus = 0
    sum_minus = 0
    for i in range(np.power(2,K)):
	binary = "{0:b}".format(i)
	binary = binary.zfill(K)
	binary = [int(d) for d in binary]
	binary = np.array(binary)
	x = (binary*2) - 1
        tmp_snr=(H.T.dot(H)).trace()/K
        H_tmp=H/np.sqrt(tmp_snr)*np.sqrt(SNR)
	y_temp = H_tmp.dot(x)
	prob = np.exp(-0.5*(np.sum(np.power(y_temp[jj] - y[jj],2) for jj in range(N))))
	for ii in range(K):
	    if x[ii] == 1:
		final_probs_one[ii]       = final_probs_one[ii]      +((1.0/np.power(2,K))*prob)
		sum_plus = sum_plus +1
	    if x[ii] == -1:
		final_probs_minus_one[ii]       = final_probs_minus_one[ii]      +((1.0/np.power(2,K))*prob)
		sum_minus = sum_minus +1
    for ii in range(K):
	norm = final_probs_one[ii] + final_probs_minus_one[ii] 
	final_probs_one[ii] = final_probs_one[ii] / norm
	final_probs_minus_one[ii] = final_probs_minus_one[ii] / norm

    return final_probs_one,final_probs_minus_one

def validate2(y, SNR, H, K, N):
    final_probs_one = np.zeros((K))
    final_probs_minus_one = np.zeros((K))
    sum_plus = 0
    sum_minus = 0
    for i in range(np.power(2, K)):
        binary = "{0:b}".format(i)
        binary = binary.zfill(K)
        binary = [int(d) for d in binary]
        binary = np.array(binary)
        x = (binary * 2) - 1
        tmp_snr = (H.dot(H.T)).trace() / K
        H_tmp = H / np.sqrt(tmp_snr) * np.sqrt(SNR)
        y_temp = x.dot(H_tmp)
        prob = np.exp(-0.5 * (np.sum(np.power(y_temp[jj] - y[jj], 2) for jj in range(N))))
        for ii in range(K):
            if x[ii] == 1:
                final_probs_one[ii] = final_probs_one[ii] + ((1.0 / np.power(2, K)) * prob)
                sum_plus = sum_plus + 1
            if x[ii] == -1:
                final_probs_minus_one[ii] = final_probs_minus_one[ii] + ((1.0 / np.power(2, K)) * prob)
                sum_minus = sum_minus + 1
    for ii in range(K):
        norm = final_probs_one[ii] + final_probs_minus_one[ii]
        final_probs_one[ii] = final_probs_one[ii] / norm
        final_probs_minus_one[ii] = final_probs_minus_one[ii] / norm

    return final_probs_one, final_probs_minus_one


def Choose_best(M,temp_dist,temp_codes,dist_add,BestAll):
    dist_zip_code = zip(temp_dist, temp_codes)
    dist_zip_code.sort()
    codes = [x for _, x in dist_zip_code]
    dist_zip_dist_add = zip(temp_dist, dist_add)
    dist_zip_dist_add.sort()
    added = [x for _, x in dist_zip_dist_add]
    temp_dist.sort()
    if len(codes) > np.minimum(M, len(codes)):
        BestAll.append(codes[np.minimum(M,len(codes)):-1])

    return temp_dist[0:np.minimum(M,len(temp_dist))] , codes[0:np.minimum(M,len(codes))] , added[0:np.minimum(M,len(codes))],BestAll


def SD(L,z,symbols,K,M):
    e = np.zeros((M,K))
    e[:,:] = z.dot(np.linalg.pinv(L))
    u = np.zeros((len(symbols),K))
    BestM = []
    BestMDists = []
    BestAll = []

    for k in reversed(range(K)):
        temp_dist = []
        dist_add = []
        temp_codes = BestM
        new_temp_codes = []
        for i in range(len(BestM)):
            for t in range(len(symbols)):
                u[t,k] = symbols[t]
                a_temp = (e[i,k] - u[t,k])/L[k,k]
                dist_add.append(a_temp)
                temp_dist.append(BestMDists[i] + np.power(a_temp,2))
                new_temp_codes.append(temp_codes[i]+[symbols[t]])

        if len(BestM) == 0:
            for t in range(len(symbols)):
                new_temp_codes.append([symbols[t]])
                temp_b = np.power((e[0,k]- symbols[t])/L[k,k],2)
                dist_add.append(temp_b)
                temp_dist.append(temp_b)
        BestMDists,BestM,dist_add,BestAll = Choose_best(M,temp_dist,new_temp_codes,dist_add,BestAll)
        for i in range(len(BestM)):
            for j in range(k-1):
                e[i,j] = e[i,j]-dist_add[i]*L[k,j]
    return BestM,BestAll



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
K = 10
N = 20
M = 5
snrdb_low = 7.0
snrdb_high = 14.0
snr_low = 10.0 ** (snrdb_low/10.0)
snr_high = 10.0 ** (snrdb_high/10.0)
L=30
v_size = 4*K
hl_size = 12*K
startingLearningRate = 0.0003
decay_factor = 0.97
decay_step_size = 1000
train_iter = 10
train_batch_size = 500

test_iter = 10
BestM_test_iter = 10

test_batch_size = 1

LOG_LOSS = 1
res_alpha = 0.9
num_snr = 6
snrdb_low_test = 8.0
snrdb_high_test = 13.0
print(' ')
print('BPSK soft with validation + M Best Sphere decoding + TIMES')
print('K')
print(K)

print('N')
print(N)

print('M')
print(M)
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
print(startingLearningRate)
print(decay_factor)
print(decay_step_size)
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
def generate_data_iid_test(B,K,N,snr_low,snr_high):
    H_=np.random.randn(B,N,K)
    x_=np.sign(np.random.rand(B,K)-0.5)
    x_pos = np.int64(x_ > 0)
    x_neg = np.int64(x_ < 0)
    x_ind = np.zeros([B,2*K])
    x_ind[:,0::2] = x_pos
    x_ind[:,1::2] = x_neg
    y_=np.zeros([B,N])
    w=np.random.randn(B,N)
    Hy_=x_*0
    HH_=np.zeros([B,K,K])
    SNR_= np.zeros([B])
    for i in range(B):
        SNR = np.random.uniform(low=snr_low,high=snr_high)
        H=H_[i,:,:]
        tmp_snr=(H.T.dot(H)).trace()/K
        H=H/np.sqrt(tmp_snr)*np.sqrt(SNR)
        H_[i,:,:]=H
        y_[i,:]=(H.dot(x_[i,:])+w[i,:])
        Hy_[i,:]=H.T.dot(y_[i,:])
        HH_[i,:,:]=H.T.dot( H_[i,:,:])
        SNR_[i] = SNR
    return y_,H_,Hy_,HH_,x_,SNR_,x_ind

def generate_data_iid_test2(B,K,N,snr_low,snr_high):
    H_ = np.random.randn(B,K,N)
    x_ = np.sign(np.random.rand(B,K)-0.5)
    y_ = np.zeros([B,N])
    w = np.random.randn(B,N)
    Hy_ = x_*0
    HH_ = np.zeros([B,K,K])
    SNR_ = np.zeros([B])
    for i in range(B):
        SNR = np.random.uniform(low=snr_low,high=snr_high)
        H = H_[i,:,:]
        tmp_snr=(H.dot(H.T)).trace()/K
        H = H/np.sqrt(tmp_snr)*np.sqrt(SNR)
        H_[i,:,:] = H
        y_[i,:] = (x_[i,:].dot(H)+w[i,:])
        Hy_[i,:] = H.dot(y_[i,:])
        HH_[i,:,:] = ( H_[i,:,:]).dot(H.T)
	SNR_[i] = SNR
    return y_,H_,Hy_,HH_,x_,SNR_

def generate_data_train(B,K,N,snr_low,snr_high):
    H_=np.random.randn(B,N,K)
    x_=np.sign(np.random.rand(B,K)-0.5)
    x_pos = np.int64(x_ > 0)
    x_neg = np.int64(x_ < 0)
    x_ind = np.zeros([B,2*K])
    x_ind[:,0::2] = x_pos
    x_ind[:,1::2] = x_neg
    y_=np.zeros([B,N])
    w=np.random.randn(B,N)
    Hy_=x_*0
    HH_=np.zeros([B,K,K])
    SNR_= np.zeros([B])
    for i in range(B):
        SNR = np.random.uniform(low=snr_low,high=snr_high)
        H=H_[i,:,:]
        tmp_snr=(H.T.dot(H)).trace()/K
        H=H/np.sqrt(tmp_snr)*np.sqrt(SNR)
        H_[i,:,:]=H
        y_[i,:]=(H.dot(x_[i,:])+w[i,:])
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
X_IND = tf.placeholder(tf.float32,shape=[None,2*K])
HH = tf.placeholder(tf.float32,shape=[None, K , K])

batch_size = tf.shape(HY)[0]
X_LS = tf.matmul(tf.expand_dims(HY,1),tf.matrix_inverse(HH))
X_LS= tf.squeeze(X_LS,1)
loss_LS = tf.reduce_mean(tf.square(X - X_LS))
ber_LS = tf.reduce_mean(tf.cast(tf.not_equal(X,tf.sign(X_LS)), tf.float32))


S1=[]
S1.append(tf.zeros([batch_size,K]))
S2=[]
S2.append(tf.zeros([batch_size,2*K]))
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
    temp11 = tf.matmul(tf.expand_dims(S1[-1],1),HH)
    temp1.append(tf.squeeze(temp11,1))
    first.append(delta[(i-1) * 2]*HY)
    second.append(delta[(i-1) * 2 + 1]*temp1[-1])
    Z1.append(S1[-1] - (delta[(i-1) * 2]*HY) + (delta[(i-1) * 2 + 1]*temp1[-1]))
    Z.append(tf.concat([Z1[-1], V[-1]], 1))
    ZZ,Wtemp,wtemp = relu_layer(Z[-1],(1*K) + v_size , hl_size,'relu'+str(i))
    
    W11.append(Wtemp)
    w11.append(wtemp)
    ZZ1.append((ZZ))
    
    S2_temp,W22_temp,w22_temp = sign_layer(ZZ , hl_size , 2*K,'sign'+str(i))
    
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

    S3 = tf.reshape(S2[i],[batch_size,K,2])
    
    temp_0 = S3[:,:,0]
    temp_1 = S3[:,:,1]
    
    temp_2 = 1*temp_0 + (-1)*temp_1
    S1.append(temp_2)
    if LOG_LOSS == 1:
        LOSS.append(np.log(i)*tf.reduce_mean(tf.reduce_mean(tf.square(X_IND - S2[-1]),1)))
    else:
        LOSS.append(tf.reduce_mean(tf.reduce_mean(tf.square(X_IND - S2[-1]),1)))
    BER.append(tf.reduce_mean(tf.cast(tf.not_equal(X_IND,tf.round(S2[-1])), tf.float32)))
Max_Val = tf.reduce_max(S3,axis=2, keep_dims=True)
Greater = tf.greater_equal(S3,Max_Val)
BER2 = tf.round(tf.cast(Greater,tf.float32))
X_IND_RESHPAED = tf.reshape(X_IND,[batch_size,K,2])
BER3 = tf.not_equal(BER2, X_IND_RESHPAED)
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
    batch_Y, batch_H, batch_HY, batch_HH, batch_X , SNR1, BATCH_X_IND= generate_data_train(train_batch_size,K,N,snr_low,snr_high)
    train_step.run(feed_dict={HY: batch_HY, HH: batch_HH, X: batch_X, X_IND:BATCH_X_IND})

    if i % 100 == 0 :
        sys.stderr.write(str(i)+' ')
        batch_Y, batch_H, batch_HY, batch_HH, batch_X ,SNR1, BATCH_X_IND= generate_data_iid_test(train_batch_size,K,N,snr_low,snr_high)
        results = sess.run([loss_LS,LOSS[L-1],ber_LS,SER], {HY: batch_HY, HH: batch_HH, X: batch_X, X_IND:BATCH_X_IND})
        print_string = [i]+results

        print ' '.join('%s' % x for x in print_string)
        
#compare comp
batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1,x_ind = generate_data_iid_test(test_batch_size, K, N, 10,10)

batch_size = np.shape(batch_HY)[0]
X_LS_comp = np.matmul(np.expand_dims(batch_HY,1),np.linalg.inv(batch_HH))
X_LS_comp= np.squeeze(X_LS_comp,1)
loss_LS_comp = np.mean(np.square(batch_X - X_LS_comp))

ber_LS_comp = np.mean(np.not_equal(batch_X,np.sign(X_LS_comp)).astype(np.float32))

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
delta_comp = sess.run(delta)

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


for i in range(0,L-1):
    print("layer")
    print(i)
    temp1_comp = np.matmul(np.expand_dims(S1_comp[i], 1), batch_HH)
    temp1_comp = np.squeeze(temp1_comp, 1)

    Z1_comp = S1_comp[-1] -  delta_comp[(i) * 2]*batch_HY + delta_comp[(i) * 2 + 1]*temp1_comp

    Z_comp = np.concatenate((Z1_comp, V_comp[-1]), 1)

    y_temp = np.matmul(Z_comp, W1[i]) + w1[i]
    ZZ_comp = np.maximum(0 , y_temp)

    y_temp = np.matmul(ZZ_comp , W2[i]) + w2[i]

    S2_comp.append(y_temp)

    S2_comp[i+1]=(1-res_alpha)*S2_comp[i+1]+res_alpha*S2_comp[i]
    S2_comp[i+1] =  np.clip(S2_comp[i+1], 0, 1)

    y_temp = np.matmul(ZZ_comp, W3[i]) + w3[i]

    V_comp.append(y_temp)
    V_comp[i+1] = (1 - res_alpha) * V_comp[i+1] + res_alpha * V_comp[i]

    S3_comp = np.reshape(S2_comp[i+1],[batch_size,K,2])
    

    temp_0_comp = S3_comp[:,:,0]
    temp_1_comp = S3_comp[:,:,1]
    
    temp_2_comp = 1*temp_0_comp + (-1)*temp_1_comp
    S1_comp.append(temp_2_comp)

    X_IND_reshaped_comp = np.reshape(x_ind,[batch_size,2*K])
    LOSS_comp.append(np.log(i)*np.mean(np.mean(np.square(X_IND_reshaped_comp - S2_comp[-1]),1)))
    BER_comp.append(np.mean(np.not_equal(batch_X,np.sign(S1_comp[-1])).astype(np.float32)))

Max_Val_comp = np.amax(S3_comp,axis=2,keepdims =True)
Greater_comp = np.greater_equal(S3_comp,Max_Val_comp)
BER2_comp = np.round(Greater_comp.astype(np.float32))
x_ind_reshaped = np.reshape(x_ind,[batch_size,K,2])
BER3_comp = np.not_equal(BER2_comp, x_ind_reshaped)
BER4_comp = np.sum(BER3_comp.astype(np.float32),2)
BER5_comp = np.greater(BER4_comp.astype(np.float32),0)
SER_comp =  np.mean(BER5_comp)    

toc = tm.time()
time_np = (toc-tic)/test_batch_size
print('time np')
print(time_np)



print("tf ser at layer is:")
print(np.array(sess.run(SER, {HY: batch_HY, HH: batch_HH, X: batch_X, X_IND:x_ind},))    )
print("np ser is:")
print(SER_comp)

          

#Testing the trained model
avg_val_error_last_layer = np.zeros((num_snr))


snrdb_list = np.linspace(snrdb_low_test,snrdb_high_test,num_snr)
snr_list = 10.0 ** (snrdb_list/10.0)
bers = np.zeros((4,num_snr))
times = np.zeros((4,num_snr))
tmp_bers = np.zeros((4,test_iter))
tmp_times = np.zeros((4,test_iter))
stat_total = np.zeros((20,num_snr))
stat_correct = np.zeros((20,num_snr))
for j in range(num_snr):
    for jj in range(test_iter):
        print('snr:')
        print(snrdb_list[j])
        print('test iteration:')
        print(jj)
        batch_Y, batch_H, batch_HY, batch_HH, batch_X ,SNR1, BATCH_X_IND= generate_data_iid_test(test_batch_size , K,N,snr_list[j],snr_list[j])


        tmp_bers[2,jj] = np.array(sess.run(SER, {HY: batch_HY, HH: batch_HH, X: batch_X, X_IND:BATCH_X_IND}))
        
        tmp_bers[0][jj] = np.array(sess.run(ber_LS, {HY: batch_HY, HH: batch_HH, X: batch_X, X_IND:BATCH_X_IND}))
	        
        tic = tm.time()
        last_layer = np.array(sess.run(S2[-1], {HY: batch_HY, HH: batch_HH, X: batch_X, X_IND:BATCH_X_IND}))
        toc = tm.time()
        tmp_times[0][jj] =toc - tic
                 
        tic = tm.time()
        toc = tm.time()
        tmp_times[1][jj] =toc - tic
                
        tic = tm.time()
        toc = tm.time()
        tmp_times[2][jj] =toc - tic
        
        ind1_last_layer = last_layer[:,0:2*K:2]
        ind2_last_layer = last_layer[:,1:2*K:2]
        #print(ind2)
        comb = ind1_last_layer- ind2_last_layer

        comb2 = np.sign(ind1_last_layer- ind2_last_layer)
        for ii in range(test_batch_size):
            for iii in range(K):
                bucket = np.int(np.clip(np.round(comb[ii][iii]*10+10),0,19))
                stat_total[bucket][j] = stat_total[bucket][j] +1
                if (comb2[ii][iii] == batch_X[ii][iii]):
                    stat_correct[bucket][j] = stat_correct[bucket][j] + 1
        for jjj in range(test_batch_size):
            final_probs_one,final_probs_minus_one = validate(batch_Y[jjj],SNR1[jjj],batch_H[jjj],K,N,batch_X[jjj])
            avg_val_error_last_layer[j] = avg_val_error_last_layer[j] + (1.0/(test_batch_size*test_iter*K))*np.sum(np.abs(final_probs_one-ind1_last_layer[jjj]))

	    	

    bers[:,j] = np.mean(tmp_bers,1)
    times[0,j] = np.mean(tmp_times[0])/test_batch_size
    times[1,j] = np.mean(tmp_times[1])/test_batch_size
    times[2,j] = np.mean(tmp_times[2])/test_batch_size

print('snrdb_list')
print(snrdb_list)
print('bers')
print(bers)
print('times')
print(times)
print('stat_correct')
print(stat_correct)
print('stat_total')
print(stat_total)
print('ratio')
print(stat_correct/stat_total)
print('validation error')
print(avg_val_error_last_layer)
print('time np')
print(time_np)

symbols = [-1,1]
snrdb_list = np.linspace(snrdb_low_test,snrdb_high_test,num_snr)
snr_list = 10.0 ** (snrdb_list/10.0)
basic_dist = np.zeros((num_snr,K))
ext_dist = np.zeros((num_snr,K))
new_dist = np.zeros((num_snr,K))
new_ext_dist = np.zeros((num_snr,K))
bers_basic = np.zeros((num_snr))

times2= np.zeros((num_snr,1))
for j in range(num_snr):
    print(j)
    total_wrong_basic = 0.0
    batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1 = generate_data_iid_test2(BestM_test_iter, K, N, snr_list[j], snr_list[j])
    basic_dist_temp = np.zeros((K))
    ext_dist_temp = np.zeros((K))
    new_dist_temp = np.zeros((K))
    new_ext_dist_temp = np.zeros((K))
    #print(batch_Y)
    tmp_times2 =  np.zeros((BestM_test_iter,1))
    for jj in range(BestM_test_iter):
        if jj%100 == 0:
            print(jj)
        basic_prob_one = np.zeros((K))
        ext_prob_one = np.zeros((K))
        new_prob_one = np.zeros((K))
        new_ext_prob_one = np.zeros((K))

        H = batch_H[jj]
        U, s, V = np.linalg.svd(H, full_matrices=0)
        S = np.diag(s)
        Vinv = np.linalg.pinv(V)
        H1 = U.dot(S)
        q_t, r_t = np.linalg.qr(np.transpose(H1))
        q = np.transpose(q_t)
        r = np.transpose(r_t)
        y = batch_Y
        pos = np.diag(np.sign(np.diag(r)))
        final_H = r.dot(pos)

        y = batch_Y[jj]

        final_y = y.dot(np.linalg.pinv(q.dot(V))).dot(pos)
        
        tic = tm.time()
        BestM ,BestAll= SD(final_H, final_y, symbols, K, M)

        #last_layer = np.array(sess.run(S2[-1], {HY: batch_HY, HH: batch_HH, X: batch_X, X_IND:BATCH_X_IND}))
        
        BestM =  np.flip(BestM,1)
        BestAllTemp = []
        for i in range(len(BestAll)):
            for ii in range(len(BestAll[i])):
                BestAllTemp.append(np.flip(BestAll[i][ii],0))
        BestAll = BestAllTemp
        toc = tm.time()
        tmp_times2[jj] =toc - tic
        final_guess= BestM[0]

        #print(np.sum(np.not_equal(batch_X[jj],final_guess)))
        total_wrong_basic = total_wrong_basic + np.sum(np.not_equal(batch_X[jj],final_guess))

        for i in range(K):
            #print(np.sum(np.equal([a[i] for a in BestM],1))*1.00/M)
            basic_prob_one[i] = np.sum(np.equal([a[i] for a in BestM],1))*1.00/M
        #print('basic_prob_one')


        for i in range(len(BestM)):
            BestAll.append(BestM[i])
        #for the solutions in the extended version, if solution is shorter than K, extend it to a full using the LS soluiton
        LS_solution = np.sign((batch_Y[jj]).dot(batch_H[jj].T).dot(np.linalg.inv((batch_H[jj]).dot(batch_H[jj].T))))

        for i in range(len(BestAll)):
            for ii in range(K):
                if len(BestAll[i])<=ii:
                    BestAll[i] = np.append(BestAll[i],LS_solution[ii])


        for i in range(K):
            ext_prob_one[i] = np.sum(np.equal([a[i] for a in BestAll],1))*1.00/(len(BestAll))

        exp_dists_basic = np.zeros((M))
        for i in range(M):
            exp_dists_basic[i] = np.exp(-1*np.sum(np.square(y-BestM[i].dot(H))))
        sum_dists_new = np.sum(exp_dists_basic)

        for i in range(K):
            temp_sum = 0
            for ii in range(M):
                if BestM[ii][i] == 1:
                    temp_sum = temp_sum+np.exp(-1*np.sum(np.square(y-BestM[ii].dot(H))))
            new_prob_one[i] = temp_sum/sum_dists_new

        #new calculation all candidates
        tic = tm.time()
        
        exp_dists_all = np.zeros(len(BestAll))
        for i in range(len(BestAll)):
            exp_dists_all[i] = np.exp(-1 * np.sum(np.square(y - BestAll[i].dot(H))))
        sum_dists_new_all = np.sum(exp_dists_all)
        for i in range(K):
            temp_sum = 0
            for ii in range(len(BestAll)):
                if BestAll[ii][i] == 1:
                    temp_sum = temp_sum+np.exp(-1*np.sum(np.square(y-BestAll[ii].dot(H))))
            new_ext_prob_one[i] = temp_sum/sum_dists_new_all
        toc = tm.time()
        tmp_times2[jj] =tmp_times2[jj]  +toc - tic

        real_prob_one, real_prob_min_one = validate2(batch_Y[jj], SNR1[jj], batch_H[jj], K, N)

        basic_dist_temp = basic_dist_temp + np.absolute(basic_prob_one - real_prob_one)
        ext_dist_temp = ext_dist_temp + np.absolute(ext_prob_one - real_prob_one)
        new_dist_temp = new_dist_temp + np.absolute(new_prob_one - real_prob_one)

        new_ext_dist_temp = new_ext_dist_temp + np.absolute(new_ext_prob_one - real_prob_one)

    bers_basic[j] = total_wrong_basic/(K*BestM_test_iter)
    basic_dist[j] = basic_dist_temp/BestM_test_iter
    ext_dist[j] = ext_dist_temp/BestM_test_iter
    new_dist[j] = new_dist_temp/BestM_test_iter
    new_ext_dist[j] = new_ext_dist_temp/BestM_test_iter
    print(tmp_times2)
    times2[j] = np.mean(tmp_times2)

print('bers_basic')
print(bers_basic)
#print('basic_dist')
#print(basic_dist)
#print('ext_dist')
#print(ext_dist)
#print('new_dist')
#print(new_dist)
print('new_ext_dist')
print(new_ext_dist)
print('times')
print(times2)