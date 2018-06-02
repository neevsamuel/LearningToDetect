#!/usr/bin/env python
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
        H_tmp = H #/ np.sqrt(tmp_snr) * np.sqrt(SNR)
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
    # if(any(0.8>ff>0.2  for ff in final_probs_one)):
    #    print(Real_X)
    #	print(final_probs_one)
    #	print(final_probs_minus_one)
    return final_probs_one, final_probs_minus_one

sess = tf.InteractiveSession()

#parameters
#K = 20
#N = 30
snrdb_low = 7.0
snrdb_high = 14.0
snr_low = 10.0 ** (snrdb_low/10.0)
snr_high = 10.0 ** (snrdb_high/10.0)
L=90
#v_size = 2*K
#hl_size = 8*K
startingLearningRate = 0.0001
decay_factor = 0.97
decay_step_size = 1000
train_iter = 20000
train_batch_size = 5000
test_batch_size = 10
LOG_LOSS = 1
res_alpha=0.9
num_snr = 6
snrdb_low_test=8.0
snrdb_high_test=13.0

"""Data generation for train and test phases
In this example, both functions are the same.
This duplication is in order to easily allow testing cases where the test is over different distributions of data than in the training phase.
e.g. training over gaussian i.i.d. channels and testing over a specific constant channel.
currently both test and train are over i.i.d gaussian channel.
"""
def generate_data_iid_test(B,K,N,snr_low,snr_high):
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
        H = H#/np.sqrt(tmp_snr)*np.sqrt(SNR)
        H_[i,:,:] = H
        y_[i,:] = (x_[i,:].dot(H)+w[i,:]*np.sqrt(tmp_snr) / np.sqrt(SNR))
        Hy_[i,:] = H.dot(y_[i,:])
        HH_[i,:,:] = ( H_[i,:,:]).dot(H.T)
	SNR_[i] = SNR
    return y_,H_,Hy_,HH_,x_,SNR_




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
    #print('e is:')
    #print(e)
    for k in reversed(range(K)):


        temp_dist = []
        dist_add = []
        temp_codes = BestM
        new_temp_codes = []
        for i in range(len(BestM)):
 
            for t in range(len(symbols)):
                #print('current symbol is:')
                #print(symbols[t])
                u[t,k] = symbols[t]
                a_temp = (e[i,k] - u[t,k])/L[k,k]
                dist_add.append(a_temp)
                #print('a_temp')
                #print(a_temp)
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


K = 10
N = 20
M = 7
test_iter= 300

symbols = [-1,1]
snrdb_list = np.linspace(snrdb_low_test,snrdb_high_test,num_snr)
snr_list = 10.0 ** (snrdb_list/10.0)
basic_dist = np.zeros((num_snr,K))
ext_dist = np.zeros((num_snr,K))
new_dist = np.zeros((num_snr,K))
new_ext_dist = np.zeros((num_snr,K))
bers_basic = np.zeros((num_snr))

for j in range(num_snr):
    print(j)
    total_wrong_basic = 0.0
    batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1 = generate_data_iid_test(test_iter, K, N, snr_list[j], snr_list[j])
    basic_dist_temp = np.zeros((K))
    ext_dist_temp = np.zeros((K))
    new_dist_temp = np.zeros((K))
    new_ext_dist_temp = np.zeros((K))
    #print(batch_Y)
    for jj in range(test_iter):
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


        BestM ,BestAll= SD(final_H, final_y, symbols, K, M)


        BestM =  np.flip(BestM,1)
        BestAllTemp = []
        for i in range(len(BestAll)):
            for ii in range(len(BestAll[i])):
                BestAllTemp.append(np.flip(BestAll[i][ii],0))
        BestAll = BestAllTemp

        final_guess= BestM[0]

        total_wrong_basic = total_wrong_basic + np.sum(np.not_equal(batch_X[jj],final_guess))

        for i in range(K):
            basic_prob_one[i] = np.sum(np.equal([a[i] for a in BestM],1))*1.00/M


        for i in range(len(BestM)):
            BestAll.append(BestM[i])
        #for the solutions in the extended version, if solution is shorter than K, extend it to a full using the LS soluiton
        LS_solution = np.sign((batch_Y[jj]).dot(batch_H[jj].T).dot(np.linalg.inv((batch_H[jj]).dot(batch_H[jj].T))))

        for i in range(len(BestAll)):
            for ii in range(K):
                if len(BestAll[i])<=ii:
                    BestAll[i] = np.append(BestAll[i],LS_solution[ii])


        for i in range(K):
            #print(np.sum(np.equal([a[i] for a in BestM],1))*1.00/M)
            ext_prob_one[i] = np.sum(np.equal([a[i] for a in BestAll],1))*1.00/(len(BestAll))

        #new calculation M candidates
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


        real_prob_one, real_prob_min_one = validate2(batch_Y[jj], SNR1[jj], batch_H[jj], K, N)

        basic_dist_temp = basic_dist_temp + np.absolute(basic_prob_one - real_prob_one)
        ext_dist_temp = ext_dist_temp + np.absolute(ext_prob_one - real_prob_one)
        new_dist_temp = new_dist_temp + np.absolute(new_prob_one - real_prob_one)

        new_ext_dist_temp = new_ext_dist_temp + np.absolute(new_ext_prob_one - real_prob_one)

    bers_basic[j] = total_wrong_basic/(K*test_iter)
    basic_dist[j] = basic_dist_temp/test_iter
    ext_dist[j] = ext_dist_temp/test_iter
    new_dist[j] = new_dist_temp/test_iter
    new_ext_dist[j] = new_ext_dist_temp/test_iter

print('bers_basic')
print(bers_basic)
print('basic_dist')
print(basic_dist)
print('ext_dist')
print(ext_dist)
print('new_dist')
print(new_dist)
print('new_ext_dist')
print(np.mean(new_ext_dist,1))
