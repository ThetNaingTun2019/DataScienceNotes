#!/usr/bin/env python
# coding: utf-8

# In[16]:


#Import Libralies
import numpy as np
import pandas as pd
import os
import random as rn
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#plt.style.use('seaborn-whitegrid')

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import LeakyReLU
from keras.models import load_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import numba
from numba import jit,cuda


print(tf.__version__)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)
# In[3]:


# Function For Result Evaluation 
def metrics(y_true, y_pred):
    metrics = [
        r2_score(y_true=y_true, y_pred=y_pred), 
        np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred)), 
        mean_absolute_error(y_true=y_true, y_pred=y_pred)
    ]
    
    return metrics


# In[4]:


# Function For Result Visualization
def yyplot(y_true, y_pred):
    yvalues = np.concatenate([y_true.flatten(), y_pred.flatten()])
    ymin, ymax, yrange = np.amin(yvalues), np.amax(yvalues), np.ptp(yvalues)
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, s=1.0)
    plt.plot([ymin - yrange * 0.01, ymax + yrange * 0.01], [ymin - yrange * 0.01, ymax + yrange * 0.01], color="red", linestyle = "dashed")
    plt.grid(linestyle = "dashed")
    plt.xlim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.ylim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.xlabel('y: true value')
    plt.ylabel('y: predicted value')
    plt.title('True-Predicted Plot')
    plt.show()

    return fig


# In[5]:


# Function to fix random seed
def reset_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = '0'
    rn.seed(seed) # Fixing the seed of a random function
    np.random.seed(seed) # Fixing the seed of numpy
    tf.random.set_seed(seed) # Fixing the seed of tensorflow


# In[6]:


#Data Loading
import glob

folder_path=r"csv_folder"
# folder_path = input("Enter the path of the folder where the data is stored.")
# folder_path = "../../02_Operating_Data/" #CGH Updated
# folder_path example: "C:\Users\AB280\Documents\GitHub\AIO_Regression\Dataset_generate\hysys_output\lng_production"
# # If pkl format, 
# all_files = glob.glob(folder_path + "\*.pkl")
# print(all_files)
# list = []
# for file_name in all_files:
#     df_each = pd.read_pickle(file_name)
#     list.append(df_each)
# df = pd.concat(list, axis=0, ignore_index=True)
# df.head(100)

# If csv format,
all_files = glob.glob(folder_path + "\SpecPower_enthpy_based.csv") #CPW Updated
print(all_files) #To Print File Names

list = []
for file_name in all_files:
    df_each = pd.read_csv(file_name)
    list.append(df_each)
    
df = pd.concat(list, axis=0, ignore_index=True) #To combine all files
df.tail(100)


# In[7]:


df=df.dropna()
df.info()


# In[8]:


# dict_X = {
#     'Unnamed: 0':'Date',
#     '041AI1806A.PV':   'FEED_N2',
#     '041AI1806B.PV':   'FEED_C1',
#     '041AI1806C.PV':   'FEED_C2',
#     '041AI1806D.PV':   'FEED_C3',
#     '041AI1806E.PV':   'FEED_iC4',
#     '041AI1806F.PV':   'FEED_nC4',
#     '041AI1806G.PV':   'FEED_C5+',
#     '041PI1203.PV':   'MCHE_P_IN',
#     '041TI1306.PV':   'FEED_MCHE_IN_T', #10
#     '041TI1356.PV':   'FEED_MCHE_WB_OUT', 
#     '041TI1355.PV':   'FEED_MCHE_WB_OUT_2',
#     '041PIC1210.PV':   'MCHE_OUT_P',
#     '041TI1313.PV':   'MCHE_OUT',
#     '041FIC2430.PV':   'LNG_Rundown',
#     '041FQI1004.PV':   'LNG_Rundown_Mol',
#     '041FI1004.PV':   'LNG_Rundown_2',
#     '041AI1806H.PV':   'LNG_Dens',
#     '071FI1021A.PV':   'LNG_Loading',
#     '060TI4001.PV':   'AmbTemp_1', #20
#     '060TI4002.PV':   'AmbTemp_2', 
#     '091TI4002.PV':   'AmbTemp_5',
#     '051TI1401A.PV':   'AirTemp_A',
#     '051TI1401B.PV':   'AirTemp_B',
#     '051TI1401C.PV':   'AirTemp_C',
#     '051TI1401D.PV':   'AirTemp_D',
#     '051TI1401E.PV':   'AirTemp_E',
#     '051PI1251.PV':   'P_HPMR',
#     '051PFIC2940.MV':   'COLD_JT',
#     '051FIC1051.MV':   'WARM_JT', #30
#     '051FI1051A.PV':   'MRV_F', 
#     '051FI1051B.PV':   'MRL_F',
#     '051AI1781A.PV':   'MR_N2',
#     '051AI1781B.PV':   'MR_C1',
#     '051AI1781C.PV':   'MR_C2',
#     '051AI1781D.PV':   'MR_C3',
#     '051TI1356.PV':   'MRV_MCHE_IN_T',
#     '051TI1361.PV':   'MRL_MCHE_IN_T',
#     '051TI1363.PV':   'MRL_MCHE_WB_OUT_T',
#     '051PI1256.PV':   'MRL_MCHE_WB_OUT_P', #40
#     '051TI1364.PV':   'MRL_MCHE_WB_IN_T', 
#     '051TI1359.PV':   'MRV_MCHE_CB_OUT',
#     '051PI1253.PV':   'MRV_MCHE_WB_OUT_P',
#     '051PI1254.PV':   'MRV_MCHE_CB_OUT_P',
#     '051TI1360.PV':   'MRV_MCHE_CB_IN',
#     '051TI1370.PV':   'MR_Return_T',
#     '051PI1260.PV':   'MR_Return_P',
#     '051PI3110.PV':   'LPMR_Suc_P',
#     '051TI3110.PV':   'LPMR_Suc_T',
#     '051PI3111.PV':   'LPMR_Dis_P', #50
#     '051TI3111.PV':   'LPMR_Dis_T', 
#     '051PI3120.PV':   'HPMR_Suc_P',
#     '051TI3120.PV':   'HPMR_Suc_T',
#     '051PI3121.PV':   'HPMR_Dis_P',
#     '051TI3121.PV':   'HPMR_Dis_T',
#     '051FI3121_N.PV':   'MR_FLOW',
#     '051SI3203.PV':   'C3_GT_Speed',
#     '051SI3113.PV':   'LPMR_GT_Speed',
#     '051SI3403.PV':   'HPMR_GT_Speed',
#     '063FI1001.PV':   'FFF_Flow', #60
#     '051FI1081.PV':   'Fuel_Flow_C3_Driver',
#     '051FI1083.PV':   'Fuel_Flow_LPMR_Driver',
#     '051FI1085.PV':   'Fuel_Flow_HPMR_Driver',
#     '051XI4055.PV':   'Helper_Motor_C3',
#     '051XI5055.PV':   'Helper_Motor_LPMR',
#     '051PI3010A.PV':   'LPC3_Suc_P',
#     '051TI3010.PV':   'LPC3_Suc_T',
#     '051FI3010_N.PV':   'LPC3_Suc_F',
#     '051PI3020.PV':   'MPC3_Suc_P',
#     '051TI3020.PV':   'MPC3_Suc_T', #70
#     '051FI3020_N.PV':   'MPC3_Suc_F',
#     '051PI3030.PV':   'HPC3_Suc_P',
#     '051TI3030.PV':   'HPC3_Suc_T',
#     '051FI3030_N.PV':   'HPC3_Suc_F',
#     '051PI3041.PV':   'C3_Comp_Dis_P',
#     '051TI3041.PV':   'C3_Comp_Dis_T',
#     '051FI3041_N.PV':   'C3_FLOW',
#     '051TI1307.PV':   'C3_CON_OUT',
#     '051TI1311.PV':   'C3_SUB_OUT',
#     '051TI1351.PV':   'LPMR_OUT', #80
#     '051TI1352.PV':   'HPMR_OUT', #81
# }

# col_names=df.columns.tolist()
# new_names=[dict_X.get(each, each) for each in col_names]
# df.columns=new_names


# In[9]:


df=df.dropna() #Some Values might be zero and thus result in N/A.
print(df.columns)


# In[10]:


# # # SINCE IT HAS BEEN CALCULATED IN THE PREVIOUS STATE


# #As per Note-3, Holding Mode: Loading Rate ≒ 0 and Rundown Rate > 85% * Max
# df = df[df["LNG_Loading"]<=0]
# df = df[df["LNG_Rundown"]>=(df["LNG_Rundown"].max()*.85)]

# #Calculations for LNG Prod & Others
# df["LNG_Prod"]=df["LNG_Rundown"] * df["LNG_Dens"]/1000

# df["MCHE_BTM_DT"]=df["FEED_MCHE_IN_T"]-df["MR_Return_T"]

# df["Fuel_WtFlow_C3_Driver"]=df["Fuel_Flow_C3_Driver"]  / 22.414 * 17.44

# df["C3_GT_Power"]=df["Fuel_WtFlow_C3_Driver"] * 50044 / (26.059873 * df["AmbTemp_2"] - 0.848894 * df["C3_GT_Speed"]  + 15378.1778218)

# df["Fuel_WtFlow_LPMR_Driver"]=df["Fuel_Flow_LPMR_Driver"] / 22.414 * 17.44

# df["LPMR_GT_Power"]=df["Fuel_WtFlow_LPMR_Driver"] * 50044 / (26.059873 * df["AmbTemp_2"] - 0.848894 * df["LPMR_GT_Speed"]  + 15378.1778218)

# df["Fuel_WtFlow_HPMR_Driver"]=df["Fuel_Flow_HPMR_Driver"] / 22.414 * 17.44

# df["HPMR_GT_Power"]=df["Fuel_WtFlow_HPMR_Driver"] * 50044 / (26.059873 * df["AmbTemp_2"] - 0.848894 * df["HPMR_GT_Speed"]  + 15378.1778218)

# ###

# df["C3_FLOW"] = df["C3_FLOW"]
# df["C3_Comp_Power"] = (df1["C3_Dis"] * df["C3_FLOW"] / 22.414) - ((df1["C3_Suc_LP"] * df["LPC3_Suc_F"] / 22.414) + (df1["C3_Suc_MP"] * df["MPC3_Suc_F"] / 22.414) + (df1["C3_Suc_HP"] * df["HPC3_Suc_F"] / 22.414))

# df["HPMR_Comp_Power"] = ((df1["MR_Dis_HP"] - df1["MR_Suc_HP"]) * df["MR_FLOW"] / 22.414 / 3600)
# df["LPMR_Comp_Power"] = ((df1["MR_Dis_LP"] - df1["MR_Suc_LP"]) * df["MR_FLOW"] / 22.414 / 3600)
# #SpecPower_Enthalpy can be calculated now.


# df["SpecPower"]=(df["C3_Comp_Power"]+ df["LPMR_Comp_Power"] + df["HPMR_Comp_Power"] + df["Helper_Motor_C3"] + df["Helper_Motor_LPMR"]) / df["LNG_Prod"]


# In[11]:


# explanatory variables
list_X = [
    'FEED_N2',
    'FEED_C1',
    'FEED_C2',
    'FEED_C3',
    'MCHE_P_IN',
    'MCHE_OUT',
    'AmbTemp_2',
    'P_HPMR',
#     'COLD_JT',
#     'WARM_JT',
    'MR_N2',
    'MR_C1',
    'MR_C2',
    'MR_C3',
    'MR_C4',
    'MR_Return_P',
    'MCHE_BTM_DT',
    'LPMR_Suc_P',
    'C3_CON_OUT'
]

# target variable
list_y = [
    'SpecPower_Enthalpy' 
]


# In[12]:


#Standarditizing the data 
sc_X = preprocessing.StandardScaler()
sc_y = preprocessing.StandardScaler()

X = df.loc[:, list_X].values
y = df.loc[:, list_y].values

X_std = sc_X.fit_transform(X)
y_std = sc_y.fit_transform(y)

#sc_X.fit(X=X)
#sc_y.fit(X=y)
X_std.mean(axis=0), y_std.mean(axis=0)


# In[13]:


#Data  Separating
from sklearn.utils import shuffle
X_train, X_test, y_train, y_test = train_test_split(X_std, y_std, test_size=0.10, random_state=None, shuffle=False)


# In[18]:
#reset_seed(0)

input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
tuning_records=[]
model_records=[]
#Hyperpameter 1
lyr_lst=[[10]]
# lyr_lst=[[10],[25],[50],
#      [10,5],[25,10],[50,20],
#      [25,10,5],[35,25,10],[50,35,20],
#      [25,15,10,5],[50,35,25,10],[100,50,35,20]]

#Hyperpameter 2
atvtn_lst=['relu']
#atvtn_lst=['relu','sigmoid','tanh',LeakyReLU(alpha=0.2)]

#Hyperpameter 3
#bth_lst=[256]
bth_lst=[32]

#Hyperpameter 4
lr_lst=[0.01]
#lr_lst=[0.01,0.001,0.0001]
#for z in range(1,5):

layer_qty=[10]
each_atvtn='relu'
each_opt=0.001
each_bth=256
opt = keras.optimizers.Adam(learning_rate=each_opt)

# model = Sequential()
# model.add(Dense(units=10, input_dim=input_dim, activation='relu'))
# # if len(layer_qty)>1:
# #     print("hello")
# #     for i,each in enumerate(layer_qty):
# #         try:
# #             model.add(Dense(units=layer_qty[i+1], activation=each_atvtn))
# #         except:
# #             pass
# model.add(Dense(units=output_dim))
for z in range(1,101):
    model = Sequential()
    model.add(Dense(units=10,input_dim=input_dim,activation=each_atvtn))
    # model.add(Dense(units=40, activation=each_atvtn))
    # model.add(Dense(units=15, activation=each_atvtn))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=opt, loss='mean_squared_error')

    #callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    callback=EarlyStopping(monitor="val_loss", patience=500, verbose=1, restore_best_weights=True)
    history = model.fit(x=X_train, y=y_train, batch_size=each_bth, epochs=5000, callbacks=[callback], verbose=1, validation_data=(X_test, y_test))

    for each_loss in history.history['val_loss']:
        batch_records = []
        batch_records.append(layer_qty)  # layer
        batch_records.append(each_atvtn)  # activation function
        batch_records.append(each_opt)  # lr rate
        batch_records.append(each_bth)  # batch
        batch_records.append(each_loss)
        tuning_records.append(batch_records)

    y_pred = model.predict(X_test)
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
    model_reports.to_csv("model_reports"+str(z)+".csv",index=False)

    hpara_reports=pd.DataFrame(data=tuning_records)
    hpara_reports.columns=['Layer',"Activation",'LR','Batch Size',"Loss"]
    hpara_reports.to_csv("loss_reports"+str(z)+".csv",index=False)
    model1=model
    print(z)
    model.save("Model_{}_{}.h5".format(str(z), str(mtrc[0])[:5]))
    del model
    #tf.keras.backend.clear_session()


# In[ ]:


# #Definition of neural networks and optimization functions using Sequential models
# # Fixing Random Seed
# reset_seed(0)

# # Building the Model
# input_dim = X.shape[1]
# output_dim = y.shape[1]

# model = Sequential()
# model.add(Dense(units=90,input_dim=input_dim))
# model.add(Dense(units=40, activation='relu'))
# model.add(Dense(units=15, activation='relu'))
# # model.add(Dense(units=10,input_dim=input_dim))
# # model.add(Dense(units=5, activation='tanh'))
# model.add(Dense(units=output_dim))
# #keras.layers.Dropout(0.9)
# opt = tf.keras.optimizers.Adam(learning_rate=0.0001)  # Optimizer, Setting the Learning Rate（Defaults to 0.001)
# #opt='RMSprop'
# model.compile(loss='mse', optimizer=opt,metrics=['accuracy']) 
# model.summary()


# In[ ]:


# %%time
# # Learning Execution
# batchsize = 256 # 2^10
# epoch = 5000
# es = EarlyStopping(monitor="val_loss", patience=100, verbose=1, restore_best_weights=True)
# history = model.fit(
#     x=X_train, y=y_train, batch_size=batchsize, epochs=epoch, 
#     verbose=1, validation_data=(X_test, y_test), callbacks=[es])


# In[ ]:


#Create Graph
plt.figure(figsize=(10, 4))

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss') # Validation Loss

plt.title(label='Learning curve')
plt.xlabel(xlabel='Epoch')
plt.ylabel(ylabel='MSE')

plt.legend()
plt.yscale('log')
plt.grid()
history.history.keys()


# In[ ]:


y_pred = model1.predict(X_test)
y_pred = y_pred.reshape(-1,1)

y_pred =sc_y.inverse_transform(y_pred)
y_test = sc_y.inverse_transform(y_test)
X_test =sc_X.inverse_transform(X_test)
y_train=sc_y.inverse_transform(y_train)
X_train=sc_X.inverse_transform(X_train)

mtrc = metrics(y_true=y_test, y_pred=y_pred)

print("r2:          %.4f" % mtrc[0])
print("rmse:        %.4f" % mtrc[1])
print("rmse / avg.: %.4f" % (mtrc[1] / y_test.mean()))
print("mae:         %.4f" % mtrc[2])
print("mae / avg.:  %.4f" % (mtrc[2] / y_test.mean()))


# In[ ]:


fig = yyplot(y_true = y_test, y_pred = y_pred)


# In[ ]:


# Dimension Reduction for outputs

y_train_1 = np.ravel(y_train)
y_test_1 = np.ravel(y_test)
print("No. of features: ", X_train.shape[1])
print("Training data size: ", y_train_1.shape)
print("Test data size:", y_test_1.shape)


# In[ ]:
#
#
# forest = RandomForestRegressor(n_estimators = 300, max_features =12, max_depth=10,random_state =0)
# #for each in algo_list:
# rgr = forest.fit(X_train ,y_train_1)
#
# y_train_pred = forest.predict(X_train)
# y_test_pred = forest.predict(X_test)
#
#
# print("R^2 train:",rgr.__class__.__name__ ,rgr.score(X_train, y_train_1))
# print("R^2 test:",rgr.__class__.__name__ , rgr.score(X_test, y_test_1))
#
#
# # In[ ]:
#
#
# fig = yyplot(y_true = y_test, y_pred = y_test_pred)
#
#
# # In[ ]:
#
#
# get_ipython().run_line_magic('matplotlib', 'inline')
# df_inputs = df[list_X]
# inputs_num = len(df_inputs.columns)
# importances = forest.feature_importances_
# indi = np.argsort(importances)[::-1] #descending order
# label = df_inputs.columns
#
# importances = importances[indi]
# print(importances)
# label_values=[str(round(each*100,2))+"%" for each in importances]
# #label = label[indi] #It will be in PV no.
# label=[list_X[each] for each in indi]
# res_dict=dict(zip(label, label_values))
# res_dict
#
#
# # In[ ]:
#
#
# plt.bar(range(inputs_num),importances)
# plt.xticks(range(inputs_num),label,rotation=90)
# plt.grid(True)
# plt.tick_params(labelsize=10)
# plt.tight_layout()
#
#
# # In[ ]:
#
#
# # Save the model
# model.save('LNG_SpecPower_Rev.4_Case_6.h5')
#
# from pickle import dump
# dump(sc_X, open('LNG_SpecPower_Scaler_Input_Rev.4_Case_6.pkl', 'wb'))
# dump(sc_y, open('LNG_SpecPower_Scaler_Output_Rev.4_Case_6.pkl', 'wb'))
#
#
# # <b>To further study</b>
#
# # In[ ]:
#
#
# from numba import jit, cuda
# import numpy as np
# # to measure exec time
# from timeit import default_timer as timer
#
# # normal function to run on cpu
# def func(a):
#     for i in range(10000000):
#         a[i]+= 1
#
# # function optimized to run on gpu
# @jit
# def func2(a):
#     for i in range(10000000):
#         a[i]+= 1
# if __name__=="__main__":
#     n = 10000000
#     a = np.ones(n, dtype = np.float64)
#     b = np.ones(n, dtype = np.float32)
#
#     start = timer()
#     func(a)
#     print("without GPU:", timer()-start)
#
#     start = timer()
#     func2(a)
#     print("with GPU:", timer()-start)


# In[ ]:




