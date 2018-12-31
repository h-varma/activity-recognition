import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools
import pandas as pd
import numpy as np
import os
import datetime
from datetime import datetime
from datetime import timedelta
from datetime import date

import scipy 
from scipy import optimize
import scipy.signal as signal 
import sys
import matplotlib.pyplot as plt
    
sys.path.append('../dsmuc/')
from dsmuc.custom import detect_peaks
import dsmuc.io as io
import dsmuc.preprocessing as pp
import dsmuc.features as ff
import dsmuc.custom as cs


import pytz
from azure.storage.blob import BlockBlobService
from io import StringIO
from azure.storage.blob import AppendBlobService
from azure.storage.blob import BlockBlobService
import requests
import json


import keras
from keras import backend as K
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten, Reshape, BatchNormalization
from keras.layers import LSTM,GRU
from keras.preprocessing import sequence
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Conv2D, AveragePooling1D, MaxPooling2D
from keras.models import Model

label_dict = {1:'walking',
             2:'walking upstairs',
             3:'walking downstairs',
             4:'sitting',
             5:'standing',
             6:'laying',
             7:'unknown'}
columns = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']

download_dir =  "../../data/G9_data/Downloaded"
account_name='watchstorage'
account_key='TJWcjsCs4aK9Xorw4DIAZGvKz0AFb2kvgSh49t+3nADR2usZ1ED14GLBQ/klJsSSrKykxu0ghCXn46+0bv2J8Q=='
container_name_ = 'jnj'

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Model saved
# filename = './src/finalized_model.sav'
# logreg = pickle.load(open(filename, 'rb'), encoding = 'iso-8859-1')
n_classes = 6
def CNN1(timesteps):
    # 77% accuracy
    model = Sequential()
    model.add(Conv1D(8,
                    64,
                     input_shape=(timesteps,1),
                     padding='valid',
                     strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(16,
                    16,
                     padding='valid',
                     strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(Conv1D(64,
                    8,
                     padding='valid',
                     strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(Conv1D(128,
                    4,
                     padding='valid',
                     strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(Conv1D(256,
                    2,
                     padding='valid',
                     strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    model.summary()
    
    return model
# download_dir =  "../../data/G9_data/Downloaded"


# Model saved
# filename = 'finalized_model.sav'
# logreg = pickle.load(open(filename, 'rb'), encoding = 'iso-8859-1')
model = CNN1(600)
model.load_weights('./src/weights.hdf5')
graph = tf.get_default_graph()

def time_to_str(t):
    t_woM = t.replace(microsecond=0)
    dt64 = np.datetime64(t_woM)
    a = dt64.astype('datetime64[s]')
    
    return np.datetime_as_string(a)+"Z"
def f(x):
    u, c = np.unique(x['predictions'].values, return_counts=True)
    outcome = u[np.argmax(c)]
    return outcome


# In[4]:

def do_recognition():

        ##################
    #### READ DATA ###
    ##################
    day_now=0
    day_before=1

    account_name='watchstorage'
    account_key='TJWcjsCs4aK9Xorw4DIAZGvKz0AFb2kvgSh49t+3nADR2usZ1ED14GLBQ/klJsSSrKykxu0ghCXn46+0bv2J8Q=='
    container_name_ = 'jnj'

    blob_service = BlockBlobService(account_name=account_name, account_key = account_key)


    blobs = [];blob_date = []
    generator = blob_service.list_blobs(container_name_)
    for blob in generator:
        blobs.append(blob.name)
        blob_date.append(blob.name[:10])
    blob_table = pd.DataFrame()
    blob_table['date'] = blob_date
    blob_table['blobname'] = blobs    

    today =(date.today() - timedelta(1)).strftime('%Y-%m-%d')
    yesterday = (date.today() - timedelta(2)).strftime('%Y-%m-%d')
    blob_table = blob_table[(blob_table['date']==yesterday)|(blob_table['date']==today)] 
    print(blob_table)
    
    print('Reading data for today and yesterday')
    if blob_table.shape[0]>0:
        blob_df = pd.DataFrame()
        for blobname in blob_table['blobname']:
            blob_Class = blob_service.get_blob_to_text(container_name=container_name_, blob_name = blobname)
            blob_String =blob_Class.content
            c = 0 
            for chunk in pd.read_csv(StringIO(blob_String), chunksize=10000):
                c += 1
                blob_df = blob_df.append(chunk)
                if c ==10:
                    break

        print("READ DATA FRAMES SIZE :",blob_df.shape[0])


    #################
    #################
    feature_list =  ['aoa','ate','apf','rms','std','minimax','cor','mean','min','max']
    preserved_features=['start']


    for watch_id in blob_df['id'].unique()[::-1]:
        print("Watch ", watch_id," is being processed" )
        df_temp = io.read_g9(blob_df[blob_df['id']==watch_id], sort=False)
        df_temp = df_temp.sort_index()
        print("READ DATA FRAMES SIZE AFTER CLEANING :",df_temp.shape[0])


        # Time to do analysis is specified
        start = yesterday + 'T00:00:00.0000Z'
        start_temp = np.datetime64(start)
        t = pd.Timestamp(start_temp)
        end = today + 'T00:00:00.0000Z'
        end_temp = np.datetime64(end)
        end_time = pd.Timestamp(end_temp)

        # Initialize 
       # Initialize 
        whole_window_size = timedelta(minutes = 5)
        window_size = timedelta(seconds=2)
        window_slide = timedelta(seconds=1)
        samples_count = []
        a = 0
        df_out = pd.DataFrame()
        t_start_list = []
        t_end_list = []
        outcome_list = []
        while (t + whole_window_size < end_time):
            label_list = []
            increment = 0
            X = []
            t_end5min= t + whole_window_size 
            print("doing time:",t, ' - ', t_end5min)
            t_start_list.append(time_to_str(t))
            t_end_list.append(time_to_str(t_end5min))
            if df_temp.between_time(t.to_pydatetime().time(), t_end5min.to_pydatetime().time()\
                                               ,include_start=True, include_end=False).shape[0] >= 10:


                while(t+window_slide< t_end5min):
                    t_end = t + window_size
                    snippet_df = df_temp.between_time(t.to_pydatetime().time(), t_end.to_pydatetime().time()
                                                   ,include_start=True, include_end=False)
                    if snippet_df.shape[0]>= 20:
                        X.append(snippet_df[columns].values)
                    t = t_end
            else:
                t = t_end5min

            if len(X)<=2:
                outcome = 7.0
            else:
                X_test = np.array(X)
                X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=100, dtype='float32',
                padding='pre', truncating='pre', value=0.)


                test_shape = X_test.shape
                X_test = X_test.reshape(test_shape[0], -1,1)
                with graph.as_default():
                    y_score = model.predict_proba(X_test) 
                    
                y_pred = np.argmax(pd.rolling_mean(y_score, 50,min_periods=1), axis=1) + 1 # need to add 1 to correct classes index
                u, c = np.unique(y_pred, return_counts=True)
                outcome = u[np.argmax(c)] # Majority vote


            outcome_list.append(label_dict[int(outcome)])
            out_ser = pd.Series(outcome,name=(t-whole_window_size, t) )
            df_out = df_out.append(out_ser)
            plt.plot(list(range(df_out.shape[0])), df_out[0], "*")

        plt.show()   

        # Create the json string to upload 
        dict_list = []
        for i in range(len(outcome_list)):
            payload_dict = {'address':watch_id.split("-")[2],
                 'starttime':t_start_list[i],
                 'endtime':t_end_list[i],
                 'tasklocation':'Activity',
                 'taskname':outcome_list[i],
                 'name':outcome_list[i],
                 'value':1}
            dict_list.append(payload_dict)
        payload = json.dumps(dict_list)
        url = "https://colife-dashboard.silverline.mobi/uploadActivityLabelForSmartWatch"
        headers = {
            'content-type': "application/json",
            'cache-control': "no-cache",
            'postman-token': "87b2b04f-175f-4a9b-f2c8-bf31de2cae7d"
            }
        # Send the data 
        response = requests.request("POST", url, data=payload, headers=headers)
        print(response.text)




    return True

