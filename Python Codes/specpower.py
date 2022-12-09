#!/usr/bin/env python
# coding: utf-8
import os
import pickle
import random as rn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LeakyReLU
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split 

def reset_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = '0'
    rn.seed(seed) # random関数のシードを固定
    np.random.seed(seed) # numpyのシードを固定
    tf.random.set_seed(seed) # tensorflowのシードを固定

def yyplot(y_true, y_pred):
    yvalues = np.concatenate([y_true.flatten(), y_pred.flatten()])
    ymin, ymax, yrange = np.amin(yvalues), np.amax(yvalues), np.ptp(yvalues)
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, s=1.0)
    plt.plot([ymin - yrange * 0.01, ymax + yrange * 0.01], [ymin - yrange * 0.01, ymax + yrange * 0.01], color="red", linestyle = "dashed")
    plt.xlim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.ylim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.xlabel('t: true value')
    plt.ylabel('y: predicted value')
    plt.title('True-Predicted Plot')
    plt.show()
    return fig

def metrics(y_true, y_pred):
    metrics = [r2_score(y_true=y_true, y_pred=y_pred), 
               np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred)), 
               mean_absolute_error(y_true=y_true, y_pred=y_pred)
              ]
    return metrics

reset_seed(1234)

in_path = 'specpower/gen_model1_preprocessed.pkl'
df_raw = pd.read_pickle(in_path)
df_raw.to_csv('gen_model1_preprocessed.csv',index=False)
df_raw.describe()

df = df_raw
#df = df_raw[:10000]
#df_te = df_raw[10000:]

sc_x = StandardScaler()
sc_y = StandardScaler()

# explanatory variable - given
list_u = [
    'FEED_N2',
    'FEED_C1',
    'FEED_C2',
    'FEED_C3',
    # 'FEED_iC4',
    # 'FEED_nC4',
    # 'FEED_C5+',
    'AmbTemp',
    'Cond_Out_Temp_AFC',
    # 'MCHE_IN_T',
    'WB_UA',
    'CB_UA',
]

# explanatory variable - search
list_v = [
    'HPMR_Dis_Press',
    'LPMR_Suc_Press',
    'MCHE_WB_DT',
    # 'MR_N2',
    'MR_C1',
    'MR_C2',
    'MR_C3',
]

list_x = list_u + list_v

df_x = df.loc[:, list_x]
df_x.describe()

df_y = df.loc[:, ['SpecPower']]
df_y.describe()

x = df_x.values
y = df_y.values

x_scaled = sc_x.fit_transform(x)
y_scaled = sc_y.fit_transform(y)

from pickle import dump
dump(sc_x, open('specpower/specpower_sc_x.pkl', 'wb'))
dump(sc_y, open('specpower/specpower_sc_y.pkl', 'wb'))

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.1, random_state=0)

input_dim = x_train.shape[1]
output_dim = y_train.shape[1]
tuning_records=[]
model_records=[]
#Hyperpameter 1
lyr_lst=[[10],[25],[50],
     [10,5],[25,10],[50,20],
     [25,10,5],[35,25,10],[50,35,20],
     [25,15,10,5],[50,35,25,10],[100,50,35,20]]
#lyr_lst=[[10,5]]
#Hyperpameter 2
#atvtn_lst=['relu','sigmoid','tanh',LeakyReLU(alpha=0.2)]
atvtn_lst=['tanh']
#Hyperpameter 3
#bth_lst=[32,64,128,256]
bth_lst=[32]
#Hyperpameter 4
#lr_lst=[0.01,0.001,0.0001]
lr_lst=[0.001]
for layer_qty in lyr_lst:
    for each_atvtn in atvtn_lst:
        for each_opt in lr_lst:
            opt = keras.optimizers.Adam(learning_rate=each_opt)
            for each_bth in bth_lst:
        
                model = Sequential()
                model.add(Dense(units=layer_qty[0], input_dim=input_dim, activation=each_atvtn))
                if len(layer_qty)>1:
                    for i,each in enumerate(layer_qty):
                        try:
                            model.add(Dense(units=layer_qty[i+1], activation=each_atvtn))
                        except:
                            pass
                model.add(Dense(units=output_dim))
                model.compile(optimizer=opt, loss='mean_squared_error')

                #callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
                callback=EarlyStopping(monitor="val_loss", patience=500, verbose=1, restore_best_weights=True)
                history = model.fit(x=x_train, y=y_train, batch_size=each_bth, epochs=500, callbacks=[callback], verbose=1, validation_data=(x_test, y_test))

                for each_loss in history.history['val_loss']:
                    batch_records = []
                    batch_records.append(layer_qty)  # layer
                    batch_records.append(each_atvtn)  # activation function
                    batch_records.append(each_opt)  # lr rate
                    batch_records.append(each_bth)  # batch
                    batch_records.append(each_loss)
                    tuning_records.append(batch_records)

                y_pred = model.predict(x_test)
                y_pred_inv = sc_y.inverse_transform(y_pred)
                y_test_inv = sc_y.inverse_transform(y_test)
                each_model=[]
                mtrc = metrics(y_true=y_test_inv, y_pred=y_pred_inv)
                each_model.append(layer_qty)#layer
                each_model.append(each_atvtn)#activation function
                each_model.append(each_opt)#lr rate
                each_model.append(each_bth)#batch
                each_model.append(mtrc[0])#r2
                each_model.append(mtrc[1])#rmse
                each_model.append(mtrc[1] / abs(y_test_inv.mean()))#rmse/avg
                each_model.append(mtrc[2])#mae
                each_model.append(mtrc[2] / abs(y_test_inv.mean()))#mae/avg
                model_records.append(each_model)
                model_reports=pd.DataFrame(data=model_records)
                model_reports.columns=['layer','activation','opt','batch','r2','rmse','rmse/avg','mae','mae/avg']
                model_reports.to_csv("model_reports"+str(layer_qty)+str(str(each_atvtn)[1:4])+str(each_opt)+str(each_bth)+".csv",index=False)

                hpara_reports=pd.DataFrame(data=tuning_records)
                hpara_reports.columns=['Layer',"Activation",'LR','Batch Size',"Loss"]
                hpara_reports.to_csv("loss_reports"+str(layer_qty)+str(str(each_atvtn)[1:4])+str(each_opt)+
                                     str(each_bth)+".csv",index=False)
                del model
                #print(model.layers)
print('Done')
# fig = plt.figure(figsize=(12, 9))
# plt.plot(history.history['loss'],"-",label="loss",)
# plt.plot(history.history['val_loss'],"-",label="val_loss")
# plt.title('model loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(loc='best')
# plt.show()
#
# y_pred = model.predict(x_test)
# y_pred_inv = sc_y.inverse_transform(y_pred)
# y_test_inv = sc_y.inverse_transform(y_test)
#
# mtrc = metrics(y_true=y_test_inv, y_pred=y_pred_inv)
# print("r2       : %.4f" % mtrc[0])
# print("rmse     : %.4f" % mtrc[1])
# print("rmse/avg.: %.6f" % (mtrc[1] / abs(y_test_inv.mean())))
# print("mae      : %.4f" % mtrc[2])
# print("mae/avg. : %.6f" % (mtrc[2] / abs(y_test_inv.mean())))
#
# fig = yyplot(y_true=y_test_inv, y_pred=y_pred_inv)