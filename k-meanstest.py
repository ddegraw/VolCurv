# -*- coding: utf-8 -*-

import sys
sys.path.append('R:\\Users\\4126694\\Python\\Modules')
#from blpfunctions import blpfunctions as blp
import blpfunctions as blp
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.gridspec as gridspec
from dateutil import rrule
from pandas.tseries.offsets import *
from holidays_jp import CountryHolidays
import dateutil.rrule as RR

def bbg_volcurve(ind, event, edate, numdays, interval,fld_lst):
    sec_list = blp.get_index(ind)
    volcurves = pd.DataFrame()
    fmt = "%Y-%m-%d" + 'T' + "%H:%M:%S"  #Assumes no milliseconds
    endDateTime = dt.datetime.strptime(edate, fmt)
    
    #skip SQ and holidays (Actually turns out cannot skip SQ days in blp call)
    day1 = dt.datetime(2017,1,1)
    sq = list(RR.rrule(RR.MONTHLY,byweekday=RR.FR,bysetpos=2,dtstart=day1,until=endDateTime))
    hols = list(zip(*CountryHolidays.get('JP', int(edate[0:4])))[0])
    skipdays = sq + hols
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
            
    return adv, volcurvesum.fillna(method='bfill'), volcurves.fillna(method='bfill')
    
def DTWDistance(s1, s2, w):
    DTW={}

    w = max(w, abs(len(s1)-len(s2)))

    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return np.sqrt(DTW[len(s1)-1, len(s2)-1])

def LB_Keogh(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):

        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2

    return np.sqrt(LB_sum)

def Euclid(s1, s2):
    dist = 0.0
    for i in range(len(s1)):
        dist += (s1[i] - s2[i])**2
    return np.sqrt(dist)

#Return centroids of clusters
def k_means_clust(data,num_clust,num_iter,w=4):
    centroid_list=random.sample(data.columns.values,num_clust)
    centroids = data[centroid_list]
    counter=0
    for n in range(num_iter):
        counter+=1
        #print "Iteration: ", counter
        assignments={}
        #assign data points to clusters
        for ind,i in enumerate(data):
            min_dist=float('inf')
            closest_clust=None
            for c_ind,j in enumerate(centroids):
                if LB_Keogh(data[i],centroids[j],5)<min_dist:
                    cur_dist=DTWDistance(data[i],centroids[j],w)
                    #cur_dist=Euclid(data[i],centroids[j])
                    if cur_dist<min_dist:
                        min_dist=cur_dist
                        closest_clust=c_ind
            
            if closest_clust in assignments:
                assignments[closest_clust].append(i)
            else:
                assignments[closest_clust]=[]

        #recalculate centroids of clusters
        for key in assignments:
            clust_sum=np.zeros(shape=len(data))
            if assignments[key]:
                for k in assignments[key]:
                    clust_sum = clust_sum + np.transpose(data[k].values)  
                centroids.columns = assignments.keys()
                centroids[key]=clust_sum/len(assignments[key])
                
    centroids.reindex(index=data.index.values) 
    
    return centroids

#Assign test data to centroids
def assignments(test, centroids, w):
    assignments={}
    dists = []
    for ind, i in enumerate(test):
        min_dist=float('inf')
        cur_dist=float('inf')
        closest_clust=None    
        for c_ind,j in enumerate(centroids):
            if LB_Keogh(test[i],centroids[j],5)<min_dist:
                cur_dist=DTWDistance(test[i],centroids[j],w)
                #cur_dist=Euclid(data[i],centroids[j])
                if cur_dist<min_dist:
                    min_dist=cur_dist
                    closest_clust=c_ind
        
        if assignments.has_key(closest_clust):       
            assignments[closest_clust].append(i)
        else:
            assignments[closest_clust]=[]
            assignments[closest_clust].append(i)
        
        dists.append((i,min_dist))
        
    return assignments, dists    
          


#create the volume curves
index = "NKY Index"
fld = ["VOLUME"]
event = ["TRADE"]
ed1 = "2017-02-28T15:00:00"
ed = "2017-02-27T15:00:00"
iv = 5
num_clust = 13
wind = 5
num_iter=25


adv20s, rawcurve, volcurve = bbg_volcurve(index,event,ed,20,iv,fld)
volcurve.pop('3103')
adv20s.pop('3103')
rawcurve.pop('3103')

test_adv20s, test_raw, test = bbg_volcurve(index,event,ed1,0,iv,fld)
test.pop('3103')
test_raw.pop('3103')    

mytarget_names = list(volcurve.columns.values)


centroids = k_means_clust(volcurve,num_clust,num_iter,wind)
asses, dists = assignments(volcurve,centroids,wind)
tasses, tdists = assignments(test,centroids,wind)

testtargetdict={}
targetdict ={}
y =[]

for targ in mytarget_names:
    for key, val in asses.iteritems():
        if targ in val:
            targetdict[targ]=key
            #y.append(key)
for targ in mytarget_names:
    for key, val in tasses.iteritems():
        if targ in val:
            testtargetdict[targ]=key



def assign(predict, centroids, w, sym, t_ind, lookback):
    min_dist=float('inf')
    cur_dist=float('inf')
    closest_clust=None
    start = 0
    if t_ind > lookback:
        start = t_ind-lookback
    else:
        start = 0
    for c_ind,j in enumerate(centroids):
        if LB_Keogh(predict[sym][start:t_ind],centroids[j][start:t_ind],3)<min_dist:
            cur_dist=DTWDistance(predict[sym][start:t_ind],centroids[j][start:t_ind],w)
            #cur_dist=Euclid(predict[sym][start:t_ind],centroids[j][start:t_ind])
            if cur_dist<min_dist:
                min_dist=cur_dist
                closest_clust=c_ind 
    return closest_clust


 #Look at the raw volume on the test day, need tot embed as a function
prediction = pd.DataFrame(0.0,index=test_raw.index.values,columns=test_raw.columns.values)
oprediction = pd.DataFrame(0.0,index=test_raw.index.values,columns=test_raw.columns.values)
error = pd.DataFrame(0.0,index=test_raw.index.values,columns=test_raw.columns.values)
centpred = pd.DataFrame(0.0,index=test_raw.index.values,columns=test_raw.columns.values)
normpred = pd.DataFrame(0.0,index=test_raw.index.values,columns=test_raw.columns.values)
cumtestraw=test_raw.cumsum()

'''
NEED TO ENFORCE THE PREDICTION TO BE MONOTONICALLY INCREASING
'''

for ind, sym in enumerate(test_raw):
    ocent = targetdict[sym] #get the initial centroid assignment 
    cent = ocent
    adv = advnew = adv20s[sym]
    sumnorm = cumvol = cumpred = cumact = 0
    scaler=1
    lookback = 6 #Use -1 if you want to compare the whole curve
    oprediction[sym] = centroids[ocent].copy()   
    prediction[sym] = oprediction[sym] * scaler * adv
    oprediction[sym] = oprediction[sym] * scaler * adv
    advcurv=rawcurve.copy()
    advcurv = advcurv/20.0
    for t_ind, t in enumerate(test_raw.index.values):
        tstart = t_ind - lookback
        centpred[sym][t] = cent
        
        scaler = cumtestraw[sym][t_ind]/oprediction[sym][t_ind]
        prediction[sym][t_ind+1:] = centroids[cent][t_ind+1:].copy()        
        
        if tstart < 0:
            prediction[sym][t_ind+1:] = prediction[sym][t_ind+1:] * scaler * adv
        else:
            #scaler = test_raw[sym][tstart:t_ind].sum()/advcurv[sym][tstart:t_ind].sum()
            prediction[sym][t_ind+1:] = prediction[sym][t_ind+1:] * scaler * adv
                               
        normpred[sym] = prediction[sym].copy()
        normpred[sym] = normpred[sym]/prediction[sym][-1]
        error[sym][t] = np.sqrt((normpred[sym][t_ind]-test[sym][t_ind])**2)
    
        if error[sym][t_ind] > 0.15:
            cent = assign(normpred,centroids,5,sym,t_ind,lookback)


for ind, sym in enumerate(centpred):
    for tind, t in enumerate(centpred.index.values):
        roid = centpred[sym][tind]
        oprediction[sym][t] = centroids[roid][tind]

'''
Compute the Errors
'''
hcerror={}
cerror={}
stockerror={}
prederror={}

#slippage vs. historical assignments
for key, val in targetdict.iteritems():
    hcerror[key] = Euclid(test[key],centroids[val])
    stockerror[key] = Euclid(test[key],volcurve[key])

#slippage based on current assignment 
for key, val in testtargetdict.iteritems():
    cerror[key] = Euclid(test[key],centroids[val])
    prederror[key] = Euclid(test[key],normpred[key])
        
errors = pd.DataFrame({'real vs. hist cent':hcerror, 'real vs hist stock':stockerror,'real vs. lookbak':cerror, 'vs. real vs. pred':prederror})

errors.describe()        