# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:28:12 2019

@author: 77078
"""

from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
from keras import regularizers
from keras.layers import LSTM
#X_train, y_train, X_val, y_val, X_test, y_test, gt_test, max_data, min_data = make_stock_data()
#------------------------------------csv数据读取--------------------------------------------
stock_dir='E://毕业设计//State-Frequency-Memory-stock-prediction-master///dataset//price_long_50//AAPL.csv'
data=pd.read_csv(open(stock_dir),header=0)
data=data[::-1]
print(data.head())
data=data['Open']
data=np.array(data)
#data=np.transpose(data)
print(data)
print(data.shape)
#X_train,X_test,y_train,y_test=[],[],[],[]
sc=MinMaxScaler()
print(data.reshape(-1,1).shape)
data=sc.fit_transform(data.reshape(-1,1))
data=np.transpose(data)
print(data.shape)
def data_to_train_test(data,n_in_steps,n_out_steps):
    X_data,y_data=[],[]
    for i in range(0,len(data[0])-n_in_steps-n_out_steps):
        X_data.append(data[0][i:i+n_in_steps])
        y_data.append(data[0][i+n_in_steps+n_out_steps-1])
    X_data=np.array(X_data)
    y_data=np.array(y_data)
    #X_data=X_data.reshape(X_train.shape[0],X_train.shape[1],1)
    #y_data=y_data.reshape(y_train.shape[0],1)
    return X_data,y_data


def train_test_split(X_data,y_data,train_percentage):
    n_train_intervals = math.ceil(X_data.shape[0] * train_percentage) 
    X_train,y_train=X_data[:n_train_intervals],y_data[:n_train_intervals]
    X_test,y_test=X_data[n_train_intervals:],y_data[n_train_intervals:]
    return X_train,y_train,X_test,y_test



X_data,y_data=data_to_train_test(data,20,30)
print(X_data.shape)
print(y_data.shape)

train_percentage=0.67
X_train,y_train,X_test,y_test=train_test_split(X_data,y_data,train_percentage)

X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)
y_train=y_train.reshape(y_train.shape[0],1)
y_test=y_test.reshape(y_test.shape[0],1)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

def create_model():
    model=Sequential()
    model.add(LSTM(10, input_shape=(X_train.shape[1], X_train.shape[2]),bias_regularizer=regularizers.l1_l2(0.001,0.001),
                   kernel_regularizer=regularizers.l1_l2(0.001,0.001)))
    model.add(Dense(1))
    model.compile(loss='mae',optimizer='adam')
    return model

model=create_model()
history=model.fit(X_train, y_train, epochs=50, batch_size=20,verbose=2,validation_data=(X_test, y_test),shuffle=False)

plt.plot(history.history["loss"], label="Train Loss (%s)" % 'Open')
plt.plot(history.history["val_loss"] , label="Test Loss (%s)" % 'Open')
plt.legend()
plt.savefig("E:/毕业设计/图片和结果/pureLSTM第三次/trainloss.png")
plt.show()


#------------------------------prediction---------------------------------------
yhat = model.predict(X_test, batch_size=50, verbose = 1)
print(yhat.shape)


inv_yhat=sc.inverse_transform(yhat)
print(y_test.shape)
inv_y=sc.inverse_transform(y_test)
rmse=math.sqrt(mean_squared_error(inv_y, inv_yhat))

avg=np.average(inv_y)
error_percentage=rmse/avg

print("")
print("Test Root Mean Square Error: %.3f" % rmse)
print("Test Average Value for %s: %.3f" % ('Open', avg))
print("Test Average Error Percentage: %.2f/100.00" % (error_percentage * 100))
        
plt.plot(inv_y, label="Actual (%s)" % 'Open')
plt.plot(inv_yhat, label="Predicted (%s)" % 'Open')
plt.legend()
plt.savefig("E:/毕业设计/图片和结果/pureLSTM第三次/testValidate.png")
plt.show()

#----------------------------------------------------------------------
print(inv_yhat.shape)

pre_result=pd.Series(inv_yhat.reshape(-1))

pre_result.to_csv('E:/毕业设计/emd+lstm/LSTM预测值.csv',index=None)
