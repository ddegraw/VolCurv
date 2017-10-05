import sys
sys.path.append('C:\\Anaconda2\\Lib\\site-packages\\blpfunctions')
#from blpfunctions import blpfunctions as blp
import blpfunctions as blp
import datetime as dt
import pandas as pd
from pandas.tseries.offsets import *
from holidays_jp import CountryHolidays
import dateutil.rrule as RR
import sys
import csv
import numpy as np
import time
import gc


def hist_stock_volcurve(ind, event, edate, numdays, interval,fld_lst):
    #sec_list = blp.get_index(ind)
    sec_list = ind
    print sec_list
    volcurves = pd.DataFrame()
    fmt = "%Y-%m-%d" + 'T' + "%H:%M:%S"  #Assumes no milliseconds
    endDateTime = dt.datetime.strptime(edate, fmt)
    #skip SQ and holidays (Actually turns out cannot skip SQ days in blp call)
    day1 = dt.datetime(int(edate[0:4]),1,1)
    sq = list(RR.rrule(RR.MONTHLY,byweekday=RR.FR,bysetpos=2,dtstart=day1,until=endDateTime))
    hols = list(zip(*CountryHolidays.get('JP', int(edate[0:4])))[0])
    skipdays = hols #+ sq if want to skip SQ days
    bday_jp = CustomBusinessDay(holidays=skipdays)
    
    startDateTime = endDateTime.replace(hour=9) - numdays*bday_jp
    numdays = pd.date_range(startDateTime, endDateTime, freq=bday_jp).nunique()
    sdate = startDateTime.strftime(fmt)
    
    for stock in sec_list:
        output=blp.get_Bars(stock, event, sdate, edate, interval, fld_lst)
        output.rename(columns={'VOLUME':stock},inplace=True)
        volcurves = volcurves.join(output,how="outer")

    #process the raw data into historical averages
    volcurves.rename(columns=lambda x: x[:4], inplace=True)
    timevect = pd.Series(volcurves.index.values)
    timeframet = timevect.to_frame()
    timeframet.columns = ['date']
    timeframet.set_index(timevect,inplace=True)
    timeframet['bucket'] = timeframet['date'].apply(lambda x: dt.datetime.strftime(x, '%H:%M:%S'))
    timeframet['date'] = timeframet['date'].apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d'))
    timeframet=timeframet.join(volcurves)
    timeframet = timeframet[timeframet.bucket != '15:10:00']
    absvolcurve = pd.pivot_table(timeframet,index='bucket',columns='date').fillna(0).T.reset_index(level=0,drop=True).iloc[:,:60].cumsum(axis=1)
    
    volcurve = absvolcurve.div(absvolcurve.sum(axis=1), axis=0).cumsum(axis=1)
            
    return absvolcurve.fillna(method='bfill'), volcurve.fillna(method='bfill')
    
def load_data(stock, seq_len, tt_split=0.1):
    num_features = len(stock.columns) # 5
    data = stock.as_matrix() 
    sequence_length = seq_len + 1 # index starting from 0
    result = []
    
    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length]) # index : index + seq_len
    
    result = np.array(result)
    row = round((1-tt_split) * result.shape[0]) # default 90% split
    train = result[:int(row), :] # 90% date, all features 
    
    x_train = train[:, :-1] 
    y_train = train[:, :-1][:,-1]
    
    x_test = result[int(row):, :-1] 
    y_test = result[int(row):, :-1][:,-1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], num_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], num_features))  

    return [x_train, y_train, x_test, y_test]
    
def proc_data(stock_list, window):      
    fld_l = ["VOLUME"]
    event_l = ["TRADE"]
    iv = 5
    ed1 = "2017-09-29T15:00:00"
    days = 59
    Train_list = []
    test_list =[]
    y_dict = {}
    y_test_dict = {}
    for ind,stock in enumerate(stock_list):
        print stock
        #abscurve, relcurve = hist_stock_volcurve([stock +' JT Equity'],event_l,ed1,days,iv,fld_l)
        abscurve, relcurve = hist_stock_volcurve([stock],event_l,ed1,days,iv,fld_l)
        X_train, y_train, X_test, y_test = load_data(relcurve, window)
        aX_train, ay_train, aX_test, ay_test = load_data(abscurve, window)
        Train_list.append(X_train)
        Train_list.append(aX_train)
        y_dict[stock] = y_train
        #y_list.append(ay_train)
        test_list.append(X_test)  
        test_list.append(aX_test) 
        y_test_dict[stock]=y_test
        #y_test_list.append(ay_test)
    return Train_list, y_dict, test_list, y_test_dict
    
window = 3
fld_l = ["VOLUME"]
event_l = ["TRADE"]
iv = 5
ed1 = "2017-09-28T15:00:00"
days = 57

abscurve, relcurve = hist_stock_volcurve(['6501 JP Equity'],event_l,ed1,days,iv,fld_l)
X_train, y_train, X_test, y_test = load_data(relcurve, window)
aX_train, ay_train, aX_test, ay_test = load_data(abscurve, window)