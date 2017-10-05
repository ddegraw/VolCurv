from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM, Input
from keras.optimizers import SGD
from keras.callbacks import TensorBoard

import sys
sys.path.append('C:\\Anaconda2\\Lib\\site-packages\\blpfunctions')
import blpfunctions as blp
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from pandas.tseries.offsets import *
from holidays_jp import CountryHolidays
import dateutil.rrule as RR


def volcurv(stock, event, edate, numdays, interval, fld_lst):

    volcurves = pd.DataFrame()
    fmt = "%Y-%m-%d" + 'T' + "%H:%M:%S"  #Assumes no milliseconds
    endDateTime = dt.datetime.strptime(edate, fmt)
    #skip SQ and holidays (Actually turns out cannot skip SQ days in blp call)
    day1 = dt.datetime( int(edate[0:4]),1,1)
    sq = list(RR.rrule(RR.MONTHLY,byweekday=RR.FR,bysetpos=2,dtstart=day1,until=endDateTime))
    hols = list(zip(*CountryHolidays.get('JP', int(edate[0:4])))[0])
    skipdays = hols + sq #if want to skip SQ days
    bday_jp = CustomBusinessDay(holidays=skipdays)
    
    
    for i in range(numdays):
        endDateTime = dt.datetime.strptime(edate, fmt)
        startDateTime = endDateTime.replace(hour=9) - (i+1)*bday_jp
        endDateTime = startDateTime.replace(hour=15)
        sdate = startDateTime.strftime(fmt)
        endate = endDateTime.strftime(fmt)
    
        output=blp.get_Bars(stock, event, sdate, endate, interval, fld_lst)
        output.rename(columns={'VOLUME':sdate},inplace=True)
        volcurves = volcurves.join(output,how="outer")

    #process the raw data into historical averages
    volcurves.rename(columns=lambda x: x[:10], inplace=True)
    timevect = pd.Series(volcurves.index.values)
    timeframet = timevect.to_frame()
    timeframet.columns =['date']
    timeframet.set_index(timevect,inplace="True")
    timeframet['bucket'] = timeframet['date'].apply(lambda x: dt.datetime.strftime(x, '%H:%M:%S'))
    timeframet=timeframet.join(volcurves)
    volcurvesum=timeframet.groupby(['bucket']).sum()
    volcurves = volcurvesum / volcurvesum.sum()
    #volcurves = volcurves.cumsum()
    volcurves = volcurves.interpolate()
    volcurvesum = volcurvesum.interpolate()
    volcurvesum = volcurvesum.dropna(axis=1,how='all')
            
    return volcurvesum.fillna(method='bfill'), volcurves.fillna(method='bfill')

'''
stock = "1333 JT Equity"
fld = ["VOLUME"]
event = ["TRADE"]
ed1 = "2016-06-22T15:00:00"
ed = "2016-06-21T15:00:00"
iv = 5

rawcurve, volcurve = volcurv(stock,event,ed,55,iv,fld)
testraw, test = volcurv(stock,event,ed1,1,iv,fld)
   

mytarget_names = list(volcurve.columns.values)
'''

max_features = X.shape[1]
maxlen = X.shape[0]

encoding_dim = 40  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(61,))

# "encoded" is the encoded representation of the input
#encoded = Dense(encoding_dim, activation='relu')(input_img)
encoded = Dense(61, activation='relu')(input_img)
#encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
#decoded = Dense(61, activation='sigmoid')(encoded)
decoded = Dense(61, activation='relu')(decoded)
decoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(61, activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)

#encoder model
encoder = Model(input=input_img, output=encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(X, X,
                nb_epoch=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test.T, x_test.T))
                
#encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test.T)
decoded_imgs = decoder.predict(encoded_imgs)

