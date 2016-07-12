import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score,roc_auc_score
from algorithms.utils import _read_split,_class_split,_one_hot
from algorithms.daego import DAEGO
from algorithms.TFMLP import MLPR


trX, teX, trY, teY = _read_split("dataset/segment.csv",read=1,oneHot=0)

#preprocessing 
scaler=MinMaxScaler(feature_range=(0, 1))
trX_scaled=scaler.fit_transform(trX)

X0,X1=_class_split(trX_scaled,trY,oneHot=0)

layer_daego=[X0.shape[1],50,100]
syn_X=DAEGO(X1,layer_daego,200,10)

X1=np.vstack((X1,syn_X))
X1=np.column_stack((X1,np.ones(X1.shape[0])))
X0=np.column_stack((X0,np.zeros(X0.shape[0])))
Xy=np.vstack((X0,X1))
np.random.shuffle(Xy)
trY=Xy[:,Xy.shape[1]-1]
# trY=_one_hot(trY)
trX=np.delete(Xy,Xy.shape[1]-1,axis=1)
layers=[trX.shape[1],50,50,50,1]
print trX.shape,"trX"
print trY.shape,"trY"

i=trX.shape[1]
o=1
h=100
layers=[i,h,h,h,o]
mlpr=MLPR(layers,maxItr = 1000, tol = 0.40, reg = 0.001, verbose = True)
trY=np.reshape(trY,(len(trY),-1))
mlpr.fit(trX, trY)
pred = mlpr.predict(teX)

t=np.arange(len(pred))

plt.semilogx(t,pred)
plt.semilogx(t,teY)
plt.show()

# print pred.shape,"pred"
# print pred
# print teY
# print teY.shape,"teY"
# print precision_score(teY,pred),"Precision Score"
# print roc_auc_score(teY,pred), "ROC_AUC"