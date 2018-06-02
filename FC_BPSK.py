#!/usr/bin/env python
#__author__ = 'neevsamuel'
import tensorflow as tf
import numpy as np
import time as tm
import sys

"""
This file is used to train and test the fullyCon architecture in the hard decision output scenario with a fixed channel.
The constellation used is BPSK and the channel is real
all parameters were optimized and trained over the 30X60 iid channel, changing the channel might require parameter tuning

Notice that the run time analysis presented in the paper was made on a numpy version of the tensorflow network.
writen by Neev Samuel based on the paper:
    "Learning to detect, Neev Samuel,Tzvi Diskin,Ami Wiesel"

contact by neev.samuel@gmail.com

"""

def generate_data(B,K,N,snr_low,snr_high,H_org):
    x_=np.sign(np.random.rand(B,K)-0.5)
    y_=np.zeros([B,N])
    w=np.random.randn(B,N)
    Hy_=x_*0
    H_ = np.zeros([B,N,K])
    HH_= np.zeros([B,K,K])
    SNR_= np.zeros([B])
    x_ind = np.zeros([B,K,2])
    for i in range(B):
        for ii in range(K):
            if x_[i][ii] == 1:
                x_ind[i][ii][0] = 1
            if x_[i][ii] == -1:
                x_ind[i][ii][1] = 1  
    for i in range(B):
        #print i
        SNR = np.random.uniform(low=snr_low,high=snr_high)
        H=H_org
        tmp_snr=(H.T.dot(H)).trace()/K
        #H=H/np.sqrt(tmp_snr)*np.sqrt(SNR)
        H_[i,:,:]=H
        y_[i,:]=(H.dot(x_[i,:])+w[i,:]*np.sqrt(tmp_snr)/np.sqrt(SNR))
        Hy_[i,:]=H.T.dot(y_[i,:])
        HH_[i,:,:]=H.T.dot( H_[i,:,:])
        SNR_[i] = SNR
    return y_,H_,Hy_,HH_,x_,SNR_,x_ind




def NN_FC(snrdb_low_test,snrdb_high_test, H,N,K,B,test_iter,train_iter,bers, num_snr):

    fc_size = 300
    np.random.seed()

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.05)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    sess = tf.InteractiveSession()

    NNinput = tf.placeholder(tf.float32, shape=[None, N], name='input')
    org_siganl = tf.placeholder(tf.float32, shape=[None, K], name='org_siganl')
    X_IND = tf.placeholder(tf.float32, shape=[None, K,2], name='org_siganl_ind')
    batchSize = tf.placeholder(tf.int32)

    W_fc1 = weight_variable([N, fc_size])
    b_fc1 = bias_variable([fc_size])
    h_fc1 = tf.nn.relu(tf.matmul(NNinput, W_fc1) + b_fc1)

    W_fc2 = weight_variable([fc_size, fc_size])
    b_fc2 = bias_variable([fc_size])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    W_fc3 = weight_variable([fc_size, fc_size])
    b_fc3 = bias_variable([fc_size])
    h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

    W_fc4 = weight_variable([fc_size, fc_size])
    b_fc4 = bias_variable([fc_size])
    h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)

    W_fc5 = weight_variable([fc_size, fc_size])
    b_fc5 = bias_variable([fc_size])
    h_fc5 = tf.nn.relu(tf.matmul(h_fc4, W_fc5) + b_fc5)

    W_fc6 = weight_variable([fc_size, 2*K])
    b_fc6 = bias_variable([2*K])
    h_fc6 = tf.matmul(h_fc5, W_fc6) + b_fc6

    output = tf.reshape(h_fc6, [batchSize,K,2])
    
    ssd = tf.reduce_sum(tf.reduce_sum(tf.square(X_IND - output)))

    
    temp_0 = output[:,:,0]
    temp_1 = output[:,:,1]
    output2 = 1*temp_0 + (-1)*temp_1
     
    rounded = tf.sign(output2)
    eq = tf.equal(rounded, org_siganl)
    eq2 = tf.reduce_sum(tf.cast(eq, tf.int32))
    
    saver = tf.train.Saver()
                    
    startingLearningRate = 0.0003
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(startingLearningRate, global_step, 1000, 0.97, staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(ssd)
    

    sess.run(tf.global_variables_initializer())

    for i in range(train_iter):
        batch_Y, batch_H, batch_HY, batch_HH, batch_X , SNR1, x_ind= generate_data(B,K,N,snrdb_low_test-1,snrdb_high_test+1,H)
        if i % 10000 == 0:

            sys.stderr.write(str(i)+' ')
            eq2.eval(feed_dict={
                NNinput: batch_Y, org_siganl: batch_X, batchSize: B, X_IND:x_ind}
            )

            train_accuracy = ssd.eval(feed_dict={
                NNinput: batch_Y, org_siganl: batch_X, batchSize: B, X_IND:x_ind}
            )
            print("step %d, training accuracy %g" % (i, train_accuracy))


        train_step.run(feed_dict={NNinput: batch_Y, org_siganl: batch_X, batchSize: B, X_IND:x_ind})

    """
    start testing our net
    """

    tmp_bers = np.zeros((1,test_iter))
    tmp_times = np.zeros((1,test_iter))
    times = np.zeros((1,1))

    #saver.restore(sess, "./FC_BPSK/FC_BPSK_model.ckpt")

    snrdb_list = np.linspace(snrdb_low_test,snrdb_high_test,num_snr)
    snr_list = 10.0 ** (snrdb_list/10.0)
    for i_snr in range (num_snr):
	Cur_SNR = snr_list[i_snr]
	print 'cur snr'
	print Cur_SNR
    	for i in range(test_iter):

		batch_Y, batch_H, batch_HY, batch_HH, batch_X , SNR1, x_ind= generate_data(B , K , N , Cur_SNR , Cur_SNR , H)
		tic = tm.time()
        	tmp_bers[0][i] =   eq2.eval(feed_dict={
        	        NNinput: batch_Y, org_siganl: batch_X, batchSize: B, X_IND:x_ind}
       		
		)
		toc = tm.time()
		tmp_times[0][i] = toc - tic

		

      		if i % 1000 == 0:
               
           		eq2.eval(feed_dict={
                	NNinput: batch_Y, org_siganl: batch_X, batchSize: B, X_IND:x_ind}
            		)
            		train_accuracy = ssd.eval(feed_dict={
                	NNinput: batch_Y, org_siganl: batch_X, batchSize: B, X_IND:x_ind}
            		)
            		print("test accuracy %g" % eq2.eval(feed_dict={
            	  	  NNinput: batch_Y, org_siganl: batch_X, batchSize: B, X_IND:x_ind}))

    	bers[0][i_snr] = np.mean(tmp_bers[0])

    
    times[0][0] = np.mean(tmp_times[0])/B
    print ('times are')
    print times
    save_path = saver.save(sess, "./FC_BPSK/FC_BPSK_model.ckpt")

    return bers/(K*B)


K = 30
N = 60
B = 1000

train_iter = 1000000

test_iter = 200

low_snr_db = 8.0
high_snr_db = 13.0
num_snr = 6


print('fully connected')
print(K)
print(N )
print(B)
print(test_iter)
print(train_iter)
print(low_snr_db)
print(high_snr_db)
print(num_snr)
bers = np.zeros((1,num_snr))

np.random.seed()

H=np.genfromtxt('toplitz055.csv', dtype=None, delimiter=',')

ans =  np.zeros((2,1))

bers = NN_FC(low_snr_db , high_snr_db , H , N , K , B , test_iter , train_iter,bers,num_snr)

bers = 1- bers

print ('ans is:')
print bers
