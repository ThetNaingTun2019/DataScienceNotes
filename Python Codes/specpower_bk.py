#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


def reset_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = '0'
    rn.seed(seed) # random関数のシードを固定
    np.random.seed(seed) # numpyのシードを固定
    tf.random.set_seed(seed) # tensorflowのシードを固定


# In[3]:


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


# In[4]:


def metrics(y_true, y_pred):
    metrics = [r2_score(y_true=y_true, y_pred=y_pred), 
               np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred)), 
               mean_absolute_error(y_true=y_true, y_pred=y_pred)
              ]
    return metrics


# In[5]:


# シードの固定
reset_seed(1234)


# In[6]:


# 学習データの読み込み
in_path = 'specpower/gen_model1_preprocessed.pkl'
df_raw = pd.read_pickle(in_path)
df_raw.describe()


# In[7]:


df = df_raw
#df = df_raw[:10000]
#df_te = df_raw[10000:]


# In[8]:


sc_x = StandardScaler()
sc_y = StandardScaler()


# In[9]:


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


# In[10]:


df_x = df.loc[:, list_x]
df_x.describe()


# In[11]:


df_y = df.loc[:, ['SpecPower']]
df_y.describe()


# In[12]:


x = df_x.values
y = df_y.values


# In[13]:


x_scaled = sc_x.fit_transform(x)
y_scaled = sc_y.fit_transform(y)


# In[14]:


# Scalerの保存
from pickle import dump
dump(sc_x, open('specpower/specpower_sc_x.pkl', 'wb'))
dump(sc_y, open('specpower/specpower_sc_y.pkl', 'wb'))


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.1, random_state=0)


# In[16]:


input_dim = x_train.shape[1]
output_dim = y_train.shape[1]
actvn='tanh'
#actvn=LeakyReLU(alpha=0.2)
#print(type(LeakyReLU(alpha=0.3)))
model = Sequential()
model.add(Dense(units=10, input_dim=input_dim,activation=actvn))

#model.add(Dense(units=15,activation=actvn))

#model.add(Dense(units=10,activation=actvn))
#model.add(LeakyReLU(alpha=0.1))
model.add(Dense(units=5,activation=actvn))
#model.add(LeakyReLU(alpha=0.1))
model.add(Dense(units=output_dim))
opt=keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer='adam', loss='mean_squared_error')
callback=EarlyStopping(monitor="val_loss", patience=50, verbose=1, restore_best_weights=True)
#callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=500, callbacks=[callback], verbose=1, validation_data=(x_test, y_test))
# In[17]:


model_path_out = 'specpower/specpower_best.h5'
mc = ModelCheckpoint(filepath=model_path_out, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
es = EarlyStopping(monitor="val_loss", patience=50, verbose=1, restore_best_weights=True)


# In[18]:


#history = model.fit(x=x_train, y=y_train, batch_size=256, epochs=500, verbose=1, callbacks=[mc, es], validation_data=(x_test, y_test))


# In[19]:


fig = plt.figure(figsize=(12, 9))
plt.plot(history.history['loss'],"-",label="loss",)
plt.plot(history.history['val_loss'],"-",label="val_loss")
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()


# In[20]:


model_best = load_model(model_path_out)


# In[21]:


y_pred = model.predict(x_test)
y_pred_inv = sc_y.inverse_transform(y_pred)
y_test_inv = sc_y.inverse_transform(y_test)


# In[22]:


# 予測精度の評価
mtrc = metrics(y_true=y_test_inv, y_pred=y_pred_inv)
print("r2       : %.4f" % mtrc[0])
print("rmse     : %.4f" % mtrc[1])
print("rmse/avg.: %.6f" % (mtrc[1] / abs(y_test_inv.mean())))
print("mae      : %.4f" % mtrc[2])
print("mae/avg. : %.6f" % (mtrc[2] / abs(y_test_inv.mean())))


# In[23]:


# テストデータの正解値と予測値のプロット
fig = yyplot(y_true=y_test_inv, y_pred=y_pred_inv)

##テストデータでの検証
# df_te_x = df_te.loc[:, list_x]
# df_te_y = df_te.loc[:, ['SpecPower']]
# x_te = df_te_x.values
# y_te = df_te_y.values
# x_te_std = sc_x.transform(x_te)
# y_te_std = model_best.predict(x_te_std)
# #y_te_std = model.predict(x_te_std)
# y_te_pred = sc_y.inverse_transform(y_te_std)# 予測精度の評価
# mtrc = metrics(y_true=y_te, y_pred=y_te_pred)
y_te = model_best.predict(x_test)
y_pred_inv = sc_y.inverse_transform(y_te)
y_test_inv = sc_y.inverse_transform(y_test)

print("r2       : %.4f" % mtrc[0])
print("rmse     : %.4f" % mtrc[1])
print("rmse/avg.: %.6f" % (mtrc[1] / abs(y_te.mean())))
print("mae      : %.4f" % mtrc[2])
print("mae/avg. : %.6f" % (mtrc[2] / abs(y_te.mean())))# テストデータの正解値と予測値のプロット
fig = yyplot(y_true=y_test_inv, y_pred=y_pred_inv)
# In[ ]: