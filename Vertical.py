# -*- coding: utf-8 -*-

#import sys
#sys.path.append('C:\\Anaconda2\\Lib\\site-packages\\blpfunctions')
#from blpfunctions 
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
    #skip SQ and holidays (Actually turns out cannot skip SQ days in blp call)
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
        dist = dist + (s1[i] - s2[i])**2
    return dist #np.sqrt(dist)

#Return centroids of clusters
def k_means_clust(data,num_clust,num_iter,w=6):
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
                if LB_Keogh(data[i],centroids[j],6)<min_dist:
                    cur_dist=DTWDistance(data[i],centroids[j],w)
                    #cur_dist=Euclid(data[i],centroids[j])
                    if cur_dist<min_dist:
                        min_dist=cur_dist
                        closest_clust=c_ind
            
            if closest_clust in assignments:
                assignments[closest_clust].append(i)
            else:
                assignments[closest_clust]=[]
        
        assignments = {i:j for i,j in assignments.items() if i is not None}
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
sym = ['6758 JP Equity']
fld = ["VOLUME"]
event = ["TRADE"]
ed1 = "2016-06-20T15:00:00"
ed = "2017-06-20T15:00:00"
iv = 5
num_clust = 15
wind = 6
num_iter=20


adv20s, rawcurve, volcurve = bbg_volcurve(sym,event,ed,1,iv,fld)
'''
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


def assign(predict, centroids, lbwind, sym, t_ind):
    #With lookback window lbwind    
    min_dist=float('inf')
    cur_dist=float('inf')
    closest_clust=None
    start = 0
    if t_ind > lbwind:
        start = t_ind-lbwind
    else:
        start = 0
    for c_ind,j in enumerate(centroids):
        if LB_Keogh(predict[sym][start:t_ind],centroids[j][start:t_ind],2)<min_dist:
            cur_dist=DTWDistance(predict[sym][start:t_ind],centroids[j][start:t_ind],lbwind)
            #cur_dist=Euclid(predict[sym][start:t_ind],centroids[j][start:t_ind])
            if cur_dist<min_dist:
                min_dist=cur_dist
                closest_clust=c_ind 
    return closest_clust

#Look at the raw volume on the test day, need tot embed as a function
prediction = pd.DataFrame(0.0,index=test_raw.index.values,columns=test_raw.columns.values)
error = pd.DataFrame(0.0,index=test_raw.index.values,columns=test_raw.columns.values)
centpred = pd.DataFrame(0.0,index=test_raw.index.values,columns=test_raw.columns.values)
cumtestraw=test_raw.cumsum()
cumerror={}

for ind, sym in enumerate(test_raw):
    ocent = targetdict[sym] #get the initial centroid assignment 
    cent = ocent
    cumerr = 0
    adv = adv20s[sym]
    sumnorm = cumvol = cumpred = cumact = 0
    scaler=1
    lookback = 5 #4 Seems to give the best results
    prediction[sym] = centroids[ocent].copy()
    for t_ind, t in enumerate(test_raw.index.values):
        #tstart = t_ind - lookback
        centpred[sym][t] = cent
        
        prediction[sym][t_ind+1:] = centroids[cent][t_ind+1:].copy()       
      
        error[sym][t] = (prediction[sym][t_ind]-test[sym][t_ind])**2
        cumerr = cumerr + error[sym][t]
        
        if error[sym][t_ind] > 0.0001:
        #if np.mod(t_ind,30) == 0:
            cent = assign(test,centroids,6,sym,t_ind) #using t_ind/2 as lookback
            #cent = assign(prediction,centroids,6,sym,t_ind)
    cumerror[sym] = cumerr
    
for ind, sym in enumerate(centpred):
    for tind, t in enumerate(centpred.index.values):
        roid = centpred[sym][tind]
        oprediction[sym][t] = centroids[roid][tind]

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
    prederror[key] = Euclid(test[key],prediction[key])
        
errors = pd.DataFrame({'real vs. hist cent':hcerror, 'real vs hist stock':stockerror,'real vs. lookbak':cerror, 'real vs. pred':prederror})


#Bit of code for comparing results with Remy
diffs = prediction.diff()
diffs.iloc[0] = prediction.iloc[0].values
diffs_test = test_raw.divide(test_raw.sum())
#diffs = diffs.divide(diffs.sum())
((diffs - diffs_test)**2).sum()



ncols = 4
nrows = 3

fig = plt.figure(figsize=(11,9))

randplotlist = random.sample(diffs.columns.values,ncols*nrows)

i=1
for value in plotlist:  #plotlist

    ax = plt.subplot(nrows,ncols,i)
    ax.grid('off')
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.plot(diffs[value].values,color='b', label='Clust')
    ax.plot(reactpred[value].values,color='g', label = 'MLP')
    ax.plot(reactact[value].values,color='r', label = 'Act')
    
    #ax.plot(prediction[value].values,color='b', label='Clust')
    #ax.plot(volcurve[value].values,color='g', label = 'Hist')
    #ax.plot(test[value].values,color='r', label = 'Act')
    
    ax.set_title(str(value),size='x-small')
    #ax.set_position([box1.x0, box1.y0, box1.width * 0.8, box1.height])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    i=i+1
#plt.figure(figsize=(11,4))
plt.subplots_adjust(hspace=0.4, vspace = 0.6)
plt.show()


fig, ax = plt.subplots()
#sns.heatmap(centpred[['2768','3101','3865','4004','4208', '4689','6674','6753','6762','7211',]],yticklabels=8)
sns.heatmap(centpred[plotlist],yticklabels=8)
plt.yticks(rotation=0) 
plt.show()
'''