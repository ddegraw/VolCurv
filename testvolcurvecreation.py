import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('R:\\Users\\4126694\\Python\\Modules')
#from blpfunctions import blpfunctions as blp
import blpfunctions as blp
from pandas.tseries.offsets import *
from holidays_jp import CountryHolidays
# get japanese holidays in 2015.


secs =['2282 JP Equity']
nky = "NKY Index"
fld_l = ["VOLUME"]
event_l = ["TRADE"]
sd = "2017-09-19T09:00:00"
ed = "2017-09-19T15:00:00"
iv = 5

sd1 = "2017-09-19T09:00:00"
ed1 = "2017-09-19T15:00:00"

def bbg_volcurve(bbgidx, event, edate, ndays, interval, fld_lst):
    #sec_list = get_index(bbgidx)
    sec_list = bbgidx
    volcurves = pd.DataFrame()
    fmt = "%Y-%m-%d" + 'T' + "%H:%M:%S"  #Assumes no milliseconds
    endDateTime = dt.datetime.strptime(edate, fmt)
    bday_jp = CustomBusinessDay(holidays=zip(*CountryHolidays.get('JP', int(edate[0:4])))[0])
    startDateTime = endDateTime.replace(hour=9) - ndays*bday_jp
    numdays =  pd.date_range(startDateTime, endDateTime, freq=bday_jp).nunique()
    sdate = startDateTime.strftime(fmt)
    for stock in sec_list:
        output=blp.get_Bars(stock, event, sdate, edate, interval, fld_lst)
        output.rename(columns={'VOLUME':stock},inplace=True)
        volcurves = volcurves.join(output,how="outer")
    #process the raw data into historical averages
    volcurves.rename(columns=lambda x: x[:4], inplace=True)
    timevect = pd.Series(volcurves.index.values)
    timeframet = timevect.to_frame()
    timeframet.columns =['date']
    timeframet.set_index(timevect,inplace="True")
    timeframet['bucket'] = timeframet['date'].apply(lambda x: dt.datetime.strftime(x, '%H:%M:%S'))
    timeframet=timeframet.join(volcurves)
    volcurvesum=timeframet.groupby(['bucket']).sum()
    adv = volcurvesum.sum()/numdays
    volcurves = volcurvesum / volcurvesum.sum()
    volcurves = volcurves.cumsum()
    volcurves = volcurves.interpolate()
    volcurvesum = volcurvesum.interpolate()
    volcurvesum = volcurvesum.dropna(axis=1,how='all')
            
    return adv, volcurvesum.fillna(method='bfill'), volcurves.fillna(method='bfill')


advs, avcurv, volcurv = bbg_volcurve(secs,event_l,ed1,1,iv,fld_l)
#test.pop('3103');