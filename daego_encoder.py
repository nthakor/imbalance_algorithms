import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
import tensorflow as tf
import math

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score

from algorithms.utils import _read_split,_class_split,_one_hot
from algorithms.daego import DAEGO
from algorithms.encoder import autoencoder,_encoder_transform

trX, teX, trY, teY = _read_split("dataset/segment.csv",read=1)

#preprocessing 
scaler=MinMaxScaler(feature_range=(0, 1))
trX_scaled=scaler.fit_transform(trX)

X0,X1=_class_split(trX_scaled,trY)
label_daf=[X0.shape[1],50,100]

Z0_=_encoder_transform(X0,label_daf,150)
Z1_=_encoder_transform(X1,label_daf,10)

label_daego=[Z1_.shape[1],80,100]
syn_Z=DAEGO(Z1_,label_daego,100,10)

Z1_1=np.vstack((Z1_,syn_Z))

label_daf.reverse()
X0_=_encoder_transform(Z0_,label_daf,150)
X1_=_encoder_transform(Z1_1,label_daf,10)


X1=np.column_stack((X1_,np.ones(X1_.shape[0])))
X0=np.column_stack((X0_,np.zeros(X0_.shape[0])))


Xy=np.vstack((X0,X1))
np.random.shuffle(Xy)
y=Xy[:,Xy.shape[1]-1]
trY=_one_hot(y)
trX=np.delete(Xy,Xy.shape[1]-1,axis=1)


teX_scaled=scaler.fit_transform(teX)
label_daf.reverse()
label_daf_test=label_daf+label_daf[::-1]
print label_daf_test,"test"
print label_daf,"label"
teX=_encoder_transform(teX_scaled,label_daf_test,10)


clf = tree.DecisionTreeRegressor()
clf = clf.fit(trX, trY)
pred=clf.predict(teX)
print f1_score(teY,pred),"F1-score"
print precision_score(teY,pred),"Precision Score"
print roc_auc_score(teY,pred), "ROC_AUC"