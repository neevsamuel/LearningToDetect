#!/usr/bin/env python
#__author__ = 'neevsamuel'
import tensorflow as tf
import numpy as np
import time as tm
import sys
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

def batch_sdr(y,H,x):
    B = np.shape(y)[0]
    ber=0.0
    for i in range(B):
        xx = sdr_ip(np.reshape(y[i],(len(y[i]),1)),H[i])
        ber+=np.mean(np.not_equal(x[i],xx))
    ber=ber/B
    return np.float32(ber)

def ampF1(s,tau):
    return np.tanh(s/tau)

def ampG1(s,tau):
    return 1-(np.tanh(s/tau) ** 2)

def amp2(y,H,N0,K,N,B):
    L = K*3
    
    beta = np.ones((B,1))*K/(0.+N)
    s = np.zeros((B,K,1))

    tau = beta*1/N0
    tau = np.expand_dims(tau,axis=2)
    y =np.expand_dims(y,axis=2)
    r=y
    for it in range(L):
        z = s+np.matmul(np.transpose(H,(0,2,1)),r)
 
        s = ampF1(z,N0*(1.0+tau))

    
        tau_new = beta/N0*np.mean(ampG1(z,N0*(1.0+tau)),1)
        tau_new = np.expand_dims(tau_new,axis=2)

        r = y - np.matmul(H,s)+tau_new/(1.0+tau)*r
        tau = tau_new
    return np.sign(s)

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

        err_amp+=(np.mean(np.not_equal(batch_X[i],xx)))/B
    return err_amp

def batch_amp2(batch_Y,batch_H,batch_X,n0,B,SNR,K,N):
    err_amp = 0.0
    xx = amp2(np.divide(batch_Y,np.sqrt(SNR)[:,None]),np.divide(batch_H,np.sqrt(SNR)[:,None,None]),n0,K,N,B)
    xx =np.squeeze(xx)
    err_amp+=np.mean(np.mean(np.not_equal(batch_X,xx)))
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

def createNoise(rows , numOfNoiseSamples):
    W = np.random.randn(rows, numOfNoiseSamples)
    #return tf.Variable(W)
    return W

def generate_data(B,K,N,snr_low,snr_high,H_org):
    #H_=np.random.randn(B,N,K)
    W_=np.zeros([B,K,K])
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
        H=H/np.sqrt(tmp_snr)*np.sqrt(SNR)
        H_[i,:,:]=H
        y_[i,:]=(H.dot(x_[i,:])+w[i,:])
        Hy_[i,:]=H.T.dot(y_[i,:])
        HH_[i,:,:]=H.T.dot( H_[i,:,:])
        SNR_[i] = SNR
    return y_,H_,Hy_,HH_,x_,SNR_,x_ind



def createSignal(signalLenght , numOfSignals):
    X = np.random.randint(2, size = (signalLenght , numOfSignals))
    X = X * 2
    X = X -1
    #return tf.Variable(tf.to_double(X))
    return X

def next_batch_random_basic(x_size, y_size, w_std, batch_size ,  H):
    Ht = np.transpose(H)
    Y_org = createSignal(y_size, batch_size)
    W_org = w_std * createNoise(x_size, batch_size)
    #Y_org = np.ones((batch_size, y_size))
    X_org = np.dot(H , Y_org) + W_org
    X_org = np.transpose(X_org)
    Y_org = np.transpose(Y_org)

    Y_org.astype(float)
    X_org.astype(float)
    batch = [X_org , Y_org]
    return batch


def NN_big_30(low_snr,high_snr, H,N,K,B,test_iter,train_iter,bers, num_snr):


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
    reg1 = tf.reduce_sum(tf.square(W_fc1))
    reg2 = tf.reduce_sum(tf.square(W_fc2))
    
    temp_0 = output[:,:,0]
    temp_1 = output[:,:,1]
    output2 = 1*temp_0 + (-1)*temp_1
                         
    startingLearningRate = 0.0003
    reg_factor = 0.000
    ssd2 = ssd + tf.multiply(reg_factor, reg1) + tf.multiply(reg_factor, reg2)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(startingLearningRate, global_step, 1000, 0.97, staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(ssd2)
    
    
    rounded = tf.sign(output2)
    eq = tf.not_equal(rounded, org_siganl)
    eq2 = tf.reduce_mean(tf.cast(eq, tf.float32))


    #rounded = tf.Print(rounded, [rounded], message="this is predicted signal after rounding")
    #ssd = tf.Print(ssd, [ssd], message="this is ssd")
    #ssd2 = tf.Print(ssd2, [ssd2], message="this is ssd2")
    #eq2 = tf.Print(eq2, [eq2], message="this is num of predicted hits")


    # testHitCount = tf.Print(testHitCount, [testHitCount], message="this is sum test hits")

    accuracy = ssd

    sess.run(tf.global_variables_initializer())

    for i in range(train_iter):
        batch_Y, batch_H, batch_HY, batch_HH, batch_X , SNR1, x_ind= generate_data(B,K,N,low_snr-1,high_snr+1,H)
        if i % 10000 == 0:

            sys.stderr.write(str(i)+' ')
            eq2.eval(feed_dict={
                NNinput: batch_Y, org_siganl: batch_X, batchSize: B, X_IND:x_ind}
            )

            train_accuracy = accuracy.eval(feed_dict={
                NNinput: batch_Y, org_siganl: batch_X, batchSize: B, X_IND:x_ind}
            )
            print("step %d, training accuracy %g" % (i, train_accuracy))


        train_step.run(feed_dict={NNinput: batch_Y, org_siganl: batch_X, batchSize: B, X_IND:x_ind})
    
    
    #time test
    tmp_times_np = np.zeros((test_iter,1))  
    W_fc1_comp = sess.run(W_fc1)
    W_fc2_comp = sess.run(W_fc2)
    W_fc3_comp = sess.run(W_fc3)
    W_fc4_comp = sess.run(W_fc4)
    W_fc5_comp = sess.run(W_fc5)
    W_fc6_comp = sess.run(W_fc6)
    b_fc1_comp = sess.run(b_fc1)
    b_fc2_comp = sess.run(b_fc2)
    b_fc3_comp = sess.run(b_fc3)
    b_fc4_comp = sess.run(b_fc4)
    b_fc5_comp = sess.run(b_fc5)
    b_fc6_comp = sess.run(b_fc6)
    for i in range(test_iter):
        batch_Y, batch_H, batch_HY, batch_HH, batch_X , SNR1, x_ind= generate_data(B , K , N , 10 , 10 , H)
   
    
        tic = tm.time()

        h_fc1_comp = np.maximum(np.matmul(batch_Y, W_fc1_comp) + b_fc1_comp,0)
        h_fc2_comp = np.maximum(np.matmul(h_fc1_comp, W_fc2_comp) + b_fc2_comp,0)
        h_fc3_comp = np.maximum(np.matmul(h_fc2_comp, W_fc3_comp) + b_fc3_comp,0)
        h_fc4_comp = np.maximum(np.matmul(h_fc3_comp, W_fc4_comp) + b_fc4_comp,0)
        h_fc5_comp = np.maximum(np.matmul(h_fc4_comp, W_fc5_comp) + b_fc5_comp,0)
        h_fc6_comp = np.matmul(h_fc5_comp, W_fc6_comp) + b_fc6_comp
                              
        output_comp = np.reshape(h_fc6_comp, [B,K,2])
        temp_0_comp = output_comp[:,:,0]
        temp_1_comp = output_comp[:,:,1]
        output2_comp = 1*temp_0_comp + (-1)*temp_1_comp
        rounded_comp = np.sign(output2_comp)
        toc = tm.time()
        eq_comp = np.not_equal(rounded_comp, batch_X)
        eq2_comp = np.sum(eq_comp.astype( np.int32))

        
        tmp_times_np[i] =   (toc-tic)
        #print('np_time')
        #print(np_time)
    np_time = np.mean(tmp_times_np)/B
    print('np_time_final')
    print(np_time)
    """
    start testing our net
    """
    tmp_bers = np.zeros((5,test_iter))

    testHitCountfull = np.zeros((6, 1))
    tmp_times = np.zeros((5,test_iter))
    times = np.zeros((5,num_snr))
    testHitCount = 0
    snr_list = np.linspace(low_snr,high_snr,num_snr)
    for i_snr in range (num_snr):
        Cur_SNR = snr_list[i_snr]
        print 'cur snr'
        print Cur_SNR
        for i in range(test_iter):
            sys.stderr.write(str(i)+' ')
            batch_Y, batch_H, batch_HY, batch_HH, batch_X , SNR1, x_ind= generate_data(B , K , N , Cur_SNR , Cur_SNR , H)
            tic = tm.time()
            tmp_bers[2][i] =   sess.run(eq2,feed_dict={NNinput: batch_Y, org_siganl: batch_X, batchSize: B, X_IND:x_ind})
            toc= tm.time()
            tmp_times[2][i] = toc - tic
           
            tic = tm.time()
            tmp_bers[3][i] = batch_sdr(batch_Y,batch_H,batch_X)
            toc = tm.time()
            tmp_times[3][i] = toc - tic
           
            tic = tm.time()
            n0 = 0.27
            tmp_bers[4][i] = batch_amp(batch_Y, batch_H, batch_X, n0, B, SNR1, K, N)
            toc = tm.time()
            tmp_times[4][i] = toc - tic
            tic = tm.time()
            tmp_bers[0][i] = batch_amp2(batch_Y, batch_H, batch_X, n0, B, SNR1, K, N)
            toc = tm.time()
            tmp_times[0][i] = toc - tic
                     
            
         
            if i % 100 == 0:
                
                eq2.eval(feed_dict={NNinput: batch_Y, org_siganl: batch_X, batchSize: B, X_IND:x_ind})
                train_accuracy = accuracy.eval(feed_dict={NNinput: batch_Y, org_siganl: batch_X, batchSize: B, X_IND:x_ind})
                print("test accuracy %g" % eq2.eval(feed_dict={NNinput: batch_Y, org_siganl: batch_X, batchSize: B, X_IND:x_ind}))

        times[0][i_snr] = np.mean(tmp_times[0])/B
        times[1][i_snr] = np.mean(tmp_times[1])/B
        times[2][i_snr] = np.mean(tmp_times[2])/B
        times[3][i_snr] = np.mean(tmp_times[3])/B
        times[4][i_snr] = np.mean(tmp_times[4])/B


        bers[0][i_snr] = np.mean(tmp_bers[0])
        bers[1][i_snr] = np.mean(tmp_bers[1])
        bers[2][i_snr] = np.mean(tmp_bers[2])
        bers[3][i_snr] = np.mean(tmp_bers[3])
        bers[4][i_snr] = np.mean(tmp_bers[4])
    
    
    print ('times are')
    print times
    return bers


K = 30
N = 60
B = 1000
test_iter = 10
train_iter = 40
low_snr_db = 8.0
high_snr_db = 13.0
num_snr = 6
low_snr = 10.0 ** (low_snr_db/10.0)
high_snr = 10.0 ** (high_snr_db/10.0)
#snrdb_list = np.linspace(snrdb_low,snrdb_high,num_snr)
#snr_list = 10.0 ** (snrdb_list/10.0)
print('fully connected')
print(K)
print(N )
print(B)
print(test_iter)
print(train_iter)
print(low_snr_db)
print(high_snr_db)
print(num_snr)
bers = np.zeros((5,num_snr))

np.random.seed()

H=np.genfromtxt('toplitz055.csv', dtype=None, delimiter=',')

ans =  np.zeros((2,1))

bers = NN_big_30(low_snr , high_snr , H , N , K , B , test_iter , train_iter,bers,num_snr)

print ('ans is:')
print bers


import numpy as np
from copy import deepcopy
import time as tm

def CreateData(K, N, SNR, B,H_const):
    print(SNR)
    H_const = H_const.T
    H_ = np.random.randn(B, K, N)
    w = np.random.randn(B, N)
    x_ = np.sign(np.random.rand(B, K) - 0.5)
    y_ = np.zeros([B, N])
    SNR_= np.zeros([B])

    for i in range(B):
        SNR = np.random.uniform(low=SNR, high=SNR)
        H = H_const
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
test_iter= [10,10,10,10,10,10]
num_snr = 6
snrdb_low_test=8.0
snrdb_high_test=13.0
snrdb_list = np.linspace(snrdb_low_test,snrdb_high_test,num_snr)
snr_list = 10.0 ** (snrdb_list/10.0)
constallation = [-1,1]


max_radius=50

for i in range(4):
    sys.stderr.write(str(i)+' ')
    max_radius = max_radius*1.2
    BERS = np.zeros([num_snr,1])
    Times = np.zeros([num_snr,1])
    for j in range(num_snr):
        temp_noises = np.zeros([test_iter[j],1])
        temp_ber = 0
        batch_Y, batch_H, batch_X ,SNR1= CreateData(K, N, snr_list[j], test_iter[j],H)
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

