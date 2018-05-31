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

def gaus(s,mean,var):
    s = np.array(s) 
    retval =np.exp(-0.5*np.real((s-mean)*np.conj(s-mean))/var**2)
    return np.real(retval)

def ampF(s,tau):
    nume   =  (1+1j)*gaus(s,(1+1j),tau) + (1-1j)*gaus(s,(1-1j),tau) + (-1+1j)*gaus(s,(-1+1j),tau) + (-1-1j)*gaus(s,(-1-1j),tau) 
    denum =          gaus(s,(1+1j),tau) +        gaus(s,(1-1j),tau) +         gaus(s,(-1+1j),tau) +         gaus(s,(-1-1j),tau) 
    retVal = (nume/(np.absolute(denum) + 0.0001))
    return retVal

def ampG(s,tau):
    second  = np.power(np.absolute(ampF(s,tau)),2)
    return 2 - second

def round1(s, K):

    retValReal = np.sign(np.real(s))
    retValIm   = np.sign(np.imag(s))
    return retValReal,retValIm


def amp(y,H,N0,N,K):
    L = K
    beta = K/(0.+N)
    s = np.zeros(K)
    tau = beta*1/N0
    r=y
    for it in range(L):

        z = s+np.dot(np.conj(H.T),r)
        s = ampF(z,N0*(1+tau))
	    #print s
        tau_new = (beta/N0)*np.mean(ampG(z,N0*(1+tau)))
        r = y - np.dot(H,s)+tau_new/(1+tau)*r
        tau = tau_new
    return round1(s,K)

def batch_amp(N,K,batch_Y,batch_H,batch_X,n0,B,SNR, x_R, x_I):
    err_amp = 0.0
    for i in range(B):
        Y = batch_Y[i][0:N] + 1j*batch_Y[i][N:2*N]
        H = batch_H[i][0:N,0:K] - 1j*batch_H[i][0:N,K:2*K]

        retValReal,retValIm = amp(Y/np.sqrt(SNR[i]),H/np.sqrt(SNR[i]),n0,N,K)
        err_amp+=(np.mean(np.logical_or(np.not_equal(x_R[i],retValReal) , np.not_equal(x_I[i],retValIm))))/(B)
    return err_amp


###start here
sess = tf.InteractiveSession()

K = 20
N = 30
train_iter = 2
snrdb_low = 7.0
snrdb_high = 14.0
snr_low = 10.0 ** (snrdb_low/10.0)
snr_high = 10.0 ** (snrdb_high/10.0)
n0=np.expand_dims(0.5,1)
res_alpha=0.9
L=30
v_size = 1*(2*K)
hl_size = 4*(2*K)
test_iter= 20
test_batch_size=100
train_batch_iter = 3000  # train batch size
num_snr = 6
snrdb_low_test=8.0
snrdb_high_test=13.0
snrdb_list = np.linspace(snrdb_low_test,snrdb_high_test,num_snr)
snr_list = 10.0 ** (snrdb_list/10.0)
startingLearningRate = 0.0003
decay_factor = 0.97
decay_step = 1000

print('QPSK one hot')
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
print(decay_step)
print(train_iter)
print(train_batch_iter)
print(test_iter)
print(test_batch_size)
print(res_alpha)
print(num_snr)
print(snrdb_low_test)
print(snrdb_high_test)

#  y = x*H' + w   H is NxK

#def generate_data(B,K,N,snr_low,snr_high):
#        X_=np.sign(np.random.randn(B,K))
#        W_=np.random.randn(B,N)
#        H_=np.random.randn(B,N,K)
#        HH_=np.zeros((B,K,K))
#        Y_=np.zeros((B,N))
#        HY_=np.zeros((B,K))
#        for b in range(B):
#            nrm=np.trace(np.dot(H_[b].T,H_[b]))/K
#            H_[b] = np.sqrt(np.random.uniform(low=snr_low,high=snr_high)/nrm)*H_[b]
#            Y_[b] = np.dot(X_[b],H_[b].T) + W_[b]
#            HY_[b] = np.dot(Y_[b],H_[b])
#            HH_[b] = np.dot(H_[b].T,H_[b])
#        return Y_,H_,HY_,HH_,X_

def generate_data_iid_test(B,K,N,snr_low,snr_high):
    H_R = np.random.randn(B,N,K)
    H_I = np.random.randn(B,N,K)
    H_  = np.zeros([B,2*N,2*K])

    #W_R = np.zeros([B,K,K])
    #W_I = np.zeros([B,K,K])

    x_R = np.sign(np.random.rand(B,K) - 0.5)
    x_I = np.sign(np.random.rand(B,K) - 0.5)
    x_  = np.concatenate((x_R , x_I) , axis = 1)

    y_  = np.zeros([B,2*N])

    w_R = np.random.randn(B,N)
    w_I = np.random.randn(B,N)
    w   = np.concatenate((w_R , w_I) , axis = 1)

    Hy_ = np.zeros([B,2*K])

    HH_ = np.zeros([B,2*K,2*K])
    SNR_ = np.zeros([B])
    x_ind = np.zeros([B,K,4])
    for i in range(B):
        for ii in range(K):
            if x_R[i,ii]==-1 and x_I[i,ii] == -1:
                x_ind[i,ii,0] =  1
            if x_R[i,ii]==-1 and x_I[i,ii] == 1:
                x_ind[i,ii,1] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == -1:
                x_ind[i,ii,2] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == 1:
                x_ind[i,ii,3] =  1

    for i in range(B):
        SNR = np.random.uniform(low=snr_low,high=snr_high)
        H   = np.concatenate((np.concatenate((H_R[i,:,:], -1*H_I[i,:,:]), axis=1) , np.concatenate((H_I[i,:,:] , H_R[i,:,:]), axis=1) ), axis=0)
        tmp_snr=(H.T.dot(H)).trace()/(2*K)
        H=H/np.sqrt(tmp_snr)*np.sqrt(SNR)
        H_[i,:,:]=H
        y_[i,:]=(H.dot(x_[i,:])+w[i,:])#*np.sqrt(tmp_snr)/np.sqrt(SNR))
        Hy_[i,:]=H.T.dot(y_[i,:])
        HH_[i,:,:]=H.T.dot( H_[i,:,:])
        SNR_[i] = SNR
    return y_,H_,Hy_,HH_,x_,SNR_, H_R, H_I, x_R, x_I, w_R, w_I, x_ind



def generate_data_train(B,K,N,snr_low,snr_high):
    H_R = np.random.randn(B,N,K)
    H_I = np.random.randn(B,N,K)
    H_  = np.zeros([B,2*N,2*K])

    #W_R = np.zeros([B,K,K])
    #W_I = np.zeros([B,K,K])

    x_R = np.sign(np.random.rand(B,K) - 0.5)
    x_I = np.sign(np.random.rand(B,K) - 0.5)
    x_  = np.concatenate((x_R , x_I) , axis = 1)

    y_  = np.zeros([B,2*N])

    w_R = np.random.randn(B,N)
    w_I = np.random.randn(B,N)
    w   = np.concatenate((w_R , w_I) , axis = 1)

    Hy_ = np.zeros([B,2*K])

    HH_ = np.zeros([B,2*K,2*K])
    SNR_ = np.zeros([B])
    x_ind = np.zeros([B,K,4])
    for i in range(B):
        for ii in range(K):
            if x_R[i,ii]==-1 and x_I[i,ii] == -1:
                x_ind[i,ii,0] =  1
            if x_R[i,ii]==-1 and x_I[i,ii] == 1:
                x_ind[i,ii,1] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == -1:
                x_ind[i,ii,2] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == 1:
                x_ind[i,ii,3] =  1
    for i in range(B):

        SNR = np.random.uniform(low=snr_low,high=snr_high)
        H   = np.concatenate((np.concatenate((H_R[i,:,:], -1*H_I[i,:,:]), axis=1) , np.concatenate((H_I[i,:,:] , H_R[i,:,:]), axis=1) ), axis=0)
        tmp_snr=(H.T.dot(H)).trace()/(2*K)
        H=H/np.sqrt(tmp_snr)*np.sqrt(SNR)
        H_[i,:,:]=H
        y_[i,:]=(H.dot(x_[i,:])+w[i,:])#*np.sqrt(tmp_snr)/np.sqrt(SNR))
        Hy_[i,:]=H.T.dot(y_[i,:])
        HH_[i,:,:]=H.T.dot( H_[i,:,:])
        SNR_[i] = SNR
    return y_,H_,Hy_,HH_,x_,SNR_, H_R, H_I, x_R, x_I, w_R, w_I,x_ind


def dfe(y,H):
    Q,R = np.linalg.qr(H)
    Qy = np.dot (Q.T,y)
    xx=np.zeros([2*K])
    for k in range(2*K-1,-1,-1):
        xx[k]=np.sign((Qy[k]-np.dot(R[k][k:],xx[k:]))/R[k][k])
    return(xx)

def batch_dfe(y,H,x,N,K):
    B = np.shape(y)[0]
    ber=0
    for i in range(B):
        xx = dfe(y[i].T,H[i])
	xx_real = xx[0:K]
	xx_imag = xx[K:2*K]
	x_real = x[i][0:K]
	x_imag = xx[K:2*K]

        ber+=np.mean(np.logical_or(np.not_equal(x_real,xx_real) , np.not_equal(x_imag,xx_imag)))
    ber=ber/B
    return np.float32(ber)


def batch_sdr(y,H,x,N,K):
    B = np.shape(y)[0]
    ber=0.0
    for i in range(B):
        xx = sdr_ip(np.reshape(y[i],(len(y[i]),1)),H[i])
        xx_real = xx[0:K]
        xx_imag = xx[K:2*K]
        x_real = x[i][0:K]
        x_imag = x[i][K:2*K]

    ber+=np.mean(np.logical_or(np.not_equal(x_real,xx_real) , np.not_equal(x_imag,xx_imag)))

    ber=ber/B
    return np.float32(ber)


def piecewise_linear_soft_sign(x):
    t = tf.Variable(0.1)
    y = -1+tf.nn.relu(x+t)/(tf.abs(t)+0.00001)-tf.nn.relu(x-t)/(tf.abs(t)+0.00001)
    return y

def affine_layer(x,input_size,output_size,Layer_num):
    W = tf.Variable(tf.random_normal([input_size, output_size], stddev=0.01))
    w = tf.Variable(tf.random_normal([1, output_size], stddev=0.01))
    #with tf.variable_scope('layer'+Layer_num):
    #    W = tf.get_variable('W', shape=[input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
    # 	print W.name
    #   w = tf.get_variable('w', shape=[1, output_size], initializer=tf.contrib.layers.xavier_initializer())
    y = tf.matmul(x, W)+w
    return y

def relu_layer(x,input_size,output_size,Layer_num):
    y = tf.nn.relu(affine_layer(x,input_size,output_size,Layer_num))
    return y

def sign_layer(x,input_size,output_size,Layer_num):
    #y = piecewise_linear_soft_sign(affine_layer(x,input_size,output_size,Layer_num))
    y = affine_layer(x,input_size,output_size,Layer_num)
    return y


HY = tf.placeholder(tf.float32,shape=[None,2*K])
X = tf.placeholder(tf.float32,shape=[None,2*K])
HH = tf.placeholder(tf.float32,shape=[None, 2*K , 2*K])
X_IND = tf.placeholder(tf.float32,shape=[None, K , 4])
batch_size = tf.shape(HY)[0]

#HY = tf.reshape(HY1,[batch_size,K])
#X = tf.reshape(X1,[batch_size,K])
#HH = tf.reshape(HH1,[batch_size,K,K])

print HH
#dHH,uHH=tf.self_adjoint_eig(HH)

X_LS = tf.matmul(tf.expand_dims(HY,1),tf.matrix_inverse(HH))
X_LS= tf.squeeze(X_LS,1)

xLS_real = tf.sign(X_LS)[:,0:K]
xLS_imag = tf.sign(X_LS)[:,K:2*K]
x_real = X[:,0:K]
x_imag = X[:,K:2*K]


loss_LS = tf.reduce_mean(tf.square(X - X_LS))
#ber_LS = tf.reduce_mean(tf.cast(tf.not_equal(X,tf.sign(X_LS)), tf.float32))
ber_LS =  tf.reduce_mean(tf.cast(tf.logical_or(tf.not_equal(x_real,xLS_real) , tf.not_equal(x_imag,xLS_imag)), tf.float32))



prefix='train_iid_test_top_055'
toptz='055'
#randomSTR = str(np.random.randint(100000))
filenamemat = 'toplitz'+toptz+'.csv'
#print filenamemat
#H_Const = np.genfromtxt(filenamemat, dtype=None, delimiter=',')
#print H_Const

S1=[]
S1.append(tf.zeros([batch_size,2*K]))
S2=[]
S2.append(tf.zeros([batch_size,4*K]))
V=[]
V.append(tf.zeros([batch_size,v_size]))
LOSS=[]
LOSS.append(tf.zeros([]))
BER=[]
BER.append(tf.zeros([]))
delta = tf.Variable(tf.zeros(L*2,1))

for i in range(1,L):
    temp1 = tf.matmul(tf.expand_dims(S1[-1],1),HH)
    temp1= tf.squeeze(temp1,1)
    Z1 = S1[-1] - delta[(i-1) * 2]*HY + delta[(i-1) * 2 + 1]*temp1
    Z = tf.concat([Z1, V[-1]], 1)
    ZZ = relu_layer(Z,(2*K) + v_size , hl_size,'relu'+str(i))
    #ZZ1 = relu_layer(Z,3*K + v_size , hl_size*2,'relu&1'+str(i))
    #ZZ = relu_layer(ZZ1,hl_size*2 , hl_size,'relu&2'+str(i))
    S2.append(sign_layer(ZZ , hl_size , 4*K,'sign'+str(i)))
    S2[i]=(1-res_alpha)*S2[i]+res_alpha*S2[i-1]
    S2[i] = tf.clip_by_value(S2[i],0,1)
    #S2[i] =    tf.clip_by_value(S2[i],0,1)
    V.append(affine_layer(ZZ , hl_size , v_size,'aff'+str(i)))
    V[i]=(1-res_alpha)*V[i]+res_alpha*V[i-1]
    
    S3 = tf.reshape(S2[i],[batch_size,K,4])
    temp_0 = S3[:,:,0]
    temp_1 = S3[:,:,1]
    temp_2 = S3[:,:,2]
    temp_3 = S3[:,:,3]
    
    S1_real = -1.0*temp_0  +\
              -1.0*temp_1  +\
              1.0*temp_2  +\
              1.0*temp_3

    S1_im =   -1.0*temp_0  +\
              1.0*temp_1  +\
              -1.0*temp_2  +\
               1.0*temp_3
    S1.append(tf.concat([S1_real, S1_im], 1))
    
    x_ind_reshaped = tf.reshape(X_IND,[batch_size,4*K])
    LOSS.append(np.log(i)*tf.reduce_mean(tf.reduce_mean(tf.square(x_ind_reshaped - S2[-1]),1)))
    #LOSS.append(tf.reduce_mean(tf.reduce_mean(tf.square(X - S[-1]),1)/tf.reduce_mean(tf.square(X - X_LS),1)))
    #BER.append(tf.reduce_mean(tf.cast(tf.not_equal(X,tf.sign(S[-1])), tf.float32)))
    BER.append(tf.reduce_mean(tf.cast(tf.logical_or(tf.not_equal(x_real,tf.sign(S1[-1][:,0:K])),tf.not_equal(x_imag,tf.sign(S1[-1][:,K:2*K]))), tf.float32)))
Max_Val = tf.reduce_max(S3,axis=2, keep_dims=True)
Greater = tf.greater_equal(S3,Max_Val)
BER2 = tf.round(tf.cast(Greater,tf.float32))

#BER2 = tf.round(S3)

BER3 = tf.not_equal(BER2, X_IND)
BER4 = tf.reduce_sum(tf.cast(BER3,tf.float32),2)
BER5 = tf.cast(tf.greater(BER4,0),tf.float32)
SER =  tf.reduce_mean(BER5)    
TOTAL_LOSS=tf.add_n(LOSS)
#TOTAL_LOSS = LOSS[-1]

saver = tf.train.Saver()

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(startingLearningRate, global_step, decay_step, decay_factor, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(TOTAL_LOSS)
#train_step = tf.train.AdamOptimizer().minimize(TOTAL_LOSS)
init_op=tf.global_variables_initializer()

#TOTAL_LOSS = tf.Print(TOTAL_LOSS, [TOTAL_LOSS], message="this is LOSS")

V = tf.Print(V, [V], message="this is V")
Z = tf.Print(Z, [Z], message="this is Z")
TOTAL_LOSS = tf.Print(TOTAL_LOSS, [TOTAL_LOSS], message="this is TOTAL_LOSS")
last_ber = 1.0

train_flag = True
#train_flag = False


if train_flag:
    sess.run(init_op)
    for i in range(train_iter): #num of train iter

        batch_Y, batch_H, batch_HY, batch_HH, batch_X , SNR1 , H_R, H_I, x_R, x_I, w_R, w_I,x_ind= generate_data_train(train_batch_iter,K,N,snr_low,snr_high)
        train_step.run(feed_dict={HY: batch_HY, HH: batch_HH, X: batch_X,X_IND:x_ind})


	
        if i % 1000== 0 :
	    TOTAL_LOSS.eval(feed_dict={
                HY: batch_HY, HH: batch_HH, X: batch_X,X_IND:x_ind}
            )
            batch_Y, batch_H, batch_HY, batch_HH, batch_X ,SNR1, H_R, H_I, x_R, x_I, w_R, w_I,x_ind= generate_data_iid_test(train_batch_iter,K,N,snr_low,snr_high)
            results = sess.run([loss_LS,LOSS[L-1],ber_LS,BER[L-1]], {HY: batch_HY, HH: batch_HH, X: batch_X,X_IND:x_ind})
            print_string = [i]+results
            print ' '.join('%s' % x for x in print_string)
            sys.stderr.write(str(i)+' ')
            with open('err_gd'+prefix+toptz+'.txt',"a") as f:
                f.write(' '.join('%s' % x for x in print_string)+'\n')             
	    if sess.run(tf.logical_or(tf.less(last_ber, 0.2*results[3]) , math.isnan(results[1])),{HY: batch_HY, HH: batch_HH, X: batch_X,X_IND:x_ind}) :
	        print 'restore model!'
	    last_ber=results[3] * 1.0
    save_path = saver.save(sess, './model.ckpt')



#sess.run(init_op)
#saver.restore(sess, './model.ckpt')

bers = np.zeros((5,num_snr))
times = np.zeros((5,num_snr))
tmp_bers = np.zeros((5,test_iter))
tmp_times = np.zeros((5,test_iter))
tmp_ber_iter=np.zeros([L,test_iter])
ber_iter=np.zeros([L,num_snr])
for j in range(num_snr):

    for jj in range(test_iter):

    	print('snr_num:')
    	print(j)
    	print(jj)

    	batch_Y, batch_H, batch_HY, batch_HH, batch_X ,SNR1, H_R, H_I, x_R, x_I, w_R, w_I,x_ind= generate_data_iid_test(test_batch_size,K,N,snr_list[j],snr_list[j])

    	tic = tm.time()
    	results = sess.run([loss_LS,LOSS[L-1],ber_LS,BER[L-1]], {HY: batch_HY, HH: batch_HH, X: batch_X,X_IND:x_ind})
        #ber_LS = tf.cast(np.mean(np.logical_or(np.not_equal(x_real, xLS_real), np.not_equal(x_imag, xLS_imag))),
        #                 tf.float32)

        tmp_ber_iter[:,jj] = np.array(sess.run(BER, {HY: batch_HY, HH: batch_HH, X: batch_X,X_IND:x_ind}))
    	results.append(batch_dfe(batch_Y,batch_H,batch_X,N,K))
    	toc = tm.time()
	tmp_times[2][jj] =toc - tic
    	
	tic = tm.time()
    	tmp_bers[1][jj]=batch_dfe(batch_Y,batch_H,batch_X,N,K)
    	toc = tm.time()
	tmp_times[1][jj] = toc - tic

    	tmp_bers[0][jj]=results[2]
    	tmp_bers[2][jj]=results[3]
    	tic = tm.time()
    	tmp_bers[3][jj]=batch_sdr(batch_Y,batch_H,batch_X,N,K)
    	toc = tm.time()
    	tmp_times[3][jj] = toc - tic

    	tic = tm.time()
	n0 = 0.35

	tmp_bers[4][jj] = batch_amp(N,K,batch_Y,batch_H,batch_X,n0,test_batch_size,SNR1, x_R, x_I)
    	toc = tm.time()
    	tmp_times[4][jj] = toc - tic

    bers[0][j] = np.mean(tmp_bers[0])
    bers[1][j] = np.mean(tmp_bers[1])
    bers[2][j] = np.mean(tmp_bers[2])
    bers[3][j] = np.mean(tmp_bers[3])
    bers[4][j] = np.mean(tmp_bers[4])
    times[1][j] = np.mean(tmp_times[1])/test_batch_size
    times[2][j] = np.mean(tmp_times[2])/test_batch_size
    times[3][j] = np.mean(tmp_times[3])/test_batch_size
    times[4][j] = np.mean(tmp_times[4])/test_batch_size
    ber_iter[:,j]=np.mean(tmp_ber_iter,1)




print('snrdb_list')
print(snrdb_list)
print('bers')
print(bers)
print('times')
print(times)
