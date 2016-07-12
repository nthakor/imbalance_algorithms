import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
import tensorflow as tf
import math

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score

from algorithms.utils import _read_split
from algorithms.daf import DAF
trX, teX, trY, teY = _read_split("dataset/segment.csv",read=1)


layer=[trX.shape[1],50,80]

def _daf_module(X,Y,layer,batch_range):
	X1=DAF(X,layer,batch_range,"sigmoid")
	X2=DAF(X,layer,batch_range,"tanh")
	Xy1=np.column_stack((X1,Y))
	Xy2=np.column_stack((X2,Y))
	Xy=np.vstack((Xy1,Xy2))
	np.random.shuffle(Xy)
	y=Xy[:,Xy.shape[1]-1]
	X=np.delete(Xy,Xy.shape[1]-1,axis=1)
	return X,y

print layer,"layer"

print "before DAF"
print trX.shape
print trY.shape
print teX.shape
print teY.shape

trX,trY=_daf_module(trX,trY,layer,10)
teX,teY=_daf_module(teX,teY,layer,10)
print "after DAF"
print trX.shape
print trY.shape
print teX.shape
print teY.shape

clf = tree.DecisionTreeRegressor()
clf = clf.fit(trX, trY)
pred=clf.predict(teX)

print pred.shape
print precision_score(teY,pred),"Precision Score"
print roc_auc_score(teY,pred), "ROC_AUC"
