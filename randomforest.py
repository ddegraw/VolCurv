# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:01:54 2016

@author: 4126694
"""

import sys
sys.path.append('R:\\Users\\4126694\\Python\\Modules')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier #use RandomForestRegressor for regression problem
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
from sklearn import metrics

X = volcurve.T

feature_names = list(X.columns.values)
mytarget_names = list(volcurve.columns.values)
targetdict ={}

indexlist =[]
imps = []
y =[]
for targ in mytarget_names:
    for key, val in asses.iteritems():
        if targ in val:
            targetdict[targ]=key
            y.append(key)

x_test = test.T
y_test = []
for targ in mytarget_names:
    for key, val in tasses.iteritems():
        if targ in val:
            testtargetdict[targ]=key
            y_test.append(key)

rf = RandomForestClassifier(n_estimators=8000, max_features = 61)
#rf = LogisticRegression()

rf.fit(X.as_matrix(), y)

probs = rf.predict_proba(x_test.as_matrix())
for feature, imp in zip(feature_names, rf.feature_importances_):
    imps.append(imp)
    indexlist.append(feature)

proba=[]
col_ind = np.argsort(probs,axis=1)[:,-1]
for i in range(len(col_ind)):
    j = col_ind[i]
    proba.append(probs[i][j])


importance = pd.DataFrame({'Importance':imps},index=indexlist)

bina = [c==d for c, d in zip (col_ind,y_test)]

fpr, tpr, _ = roc_curve(bina, proba)
roc_auc = auc(fpr, tpr)
print roc_auc
print(metrics.classification_report(col_ind, y_test))
#print(metrics.confusion_matrix(col_ind, y_test))


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

 
