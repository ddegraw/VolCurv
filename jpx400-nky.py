# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 14:06:40 2016

@author: 4126694
"""

import blpfunctions as blp
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from pandas.tseries.offsets import *
from holidays_jp import CountryHolidays
#import dateutil.rrule as RR
import sys
import csv

def bbg_volcurve(ind, event, edate, numdays, interval,fld_lst):
    sec_list = blp.get_index(ind)
    volcurves = pd.DataFrame()
    fmt = "%Y-%m-%d" + 'T' + "%H:%M:%S"  #Assumes no milliseconds
    endDateTime = dt.datetime.strptime(edate, fmt)
    #skip SQ and holidays (Actually turns out cannot skip SQ days in blp call)->which means neeed to strip out later
    #day1 = dt.datetime( int(edate[0:4]),1,1)
    #sq = list(RR.rrule(RR.MONTHLY,byweekday=RR.FR,bysetpos=2,dtstart=day1,until=endDateTime))
    hols = list(zip(*CountryHolidays.get('JP', int(edate[0:4])))[0])
    skipdays = hols #+sq if want to skip SQ days
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
    
    
index = "NKY Index"
index1 = "JPXNK400 Index"
fld = ["VOLUME"]
event = ["TRADE"]
ed1 = "2016-08-24T15:00:00"
ed = "2016-06-25T15:00:00"
iv = 5

nkyhistadv20, nkyhistrawcurve, nkyhistvolcurve = bbg_volcurve(index,event,ed,20,iv,fld)
nkyadv20, nkyrawcurve, nkyvolcurve = bbg_volcurve(index,event,ed1,0,iv,fld)

jpxhistadv20, jpxhistrawcurve, jpxhistvolcurve = bbg_volcurve(index1,event,ed,20,iv,fld)
jpxadv20, jpxrawcurve, jpxvolcurve = bbg_volcurve(index1,event,ed1,0,iv,fld)