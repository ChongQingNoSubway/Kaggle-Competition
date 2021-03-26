import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import os
cwd=os.getcwd()
from scipy.io import loadmat
import random

from sklearn.decomposition import PCA                  # PCA package
from sklearn.preprocessing import StandardScaler       # standardize data

x = loadmat(cwd+'/data.mat')
y=x.get('y')

ratio=0.7
k=5
N=len(y)

N_train=int((N-k)*ratio//1)
N_test=N-N_train-k
train_index=random.sample(range(N-k), N_train)
train_index=set(train_index)

# total number will be k (first order)+ k (second order)+C_k^2(corss second order)
N_look_ahead=2*k+int(k*(k-1)//2)
x_train=np.zeros([N_train,N_look_ahead])
y_train=np.zeros([N_train,1])

x_test=np.zeros([N_test,N_look_ahead])
y_test=np.zeros([N_test,1])
o1=0
o2=0
for i in range(0,N-k):
    if i in train_index:
        
        count=0
        # save the first order
        for j in range(0,k):
            x_train[o1][count]=y[i+j]
            count+=1
        # save the second order
        for j in range(0,k):
            x_train[o1][count]=y[i+j]**2
            count+=1
        # save the cross term
        for j1 in range(0,k):
            for j2 in range(j1+1,k):
                x_train[o1][count]=y[i+j1]*y[i+j2]
                count+=1
                
        y_train[o1][0]=y[i+k]
        o1+=1
    else:
        count=0
        # save the first order
        for j in range(0,k):
            x_test[o2][count]=y[i+j]
            count+=1
        # save the second order
        for j in range(0,k):
            x_test[o2][count]=y[i+j]**2
            count+=1
        # save the cross term
        for j1 in range(0,k):
            for j2 in range(j1+1,k):
                x_test[o2][count]=y[i+j1]*y[i+j2]
                count+=1
        y_test[o2][0]=y[i+k]
        o2+=1

sc = StandardScaler()                      # create the standard scalar
sc.fit(x_train)                            # compute the required transformation
x_train_std = sc.transform(x_train)        # apply to the training data
x_test_std = sc.transform(x_test)          # and SAME transformation of test data!!!

pca = PCA(n_components=6)                    # only keep two "best" features!
x_train_pca = pca.fit_transform(x_train_std) # apply to the train data
x_test_pca = pca.transform(x_test_std)       # do the same to the test data

model = Sequential()
model.add(Dense(6, input_dim=6,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train_pca, y_train, epochs=200, batch_size=10)
model.summary()
y_pred_train = model.predict(x_train_pca)
y_pred_test = model.predict(x_test_pca)

error_train=sum([(y_train[i]-y_pred_train[i])**2 for i in range(0,N_train)])/N_train
error_test=sum([(y_test[i]-y_pred_test[i])**2 for i in range(0,N_test)])/N_test
print(f"train error :{error_train} {'->' * 10} test error: {error_test}")