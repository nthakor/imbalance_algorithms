import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf



from algorithms.utils import _read_split,_class_split
from algorithms.daego import DAEGO
from algorithms.daf import DAF

trX, teX, trY, teY = _read_split("dataset/segment.csv",read=1)

#preprocessing 
scaler=MinMaxScaler(feature_range=(0, 1))
trX_scaled=scaler.fit_transform(trX)

X0,X1=_class_split(trX_scaled,trY)
label_daf=[X0.shape[1],30,60]

Z0_=DAF(X0,label_daf,150,"sigmoid")
Z1_=DAF(X1,label_daf,10,"sigmoid")

label_daego=[Z1_.shape[1],80,100]
syn_Z=DAEGO(Z1_,label_daego,100,10)

Z1_1=np.hstack((Z1_,syn_Z))

label_daf.reverse()
X0_=DAF(Z0_,label_daf,150,"sigmoid")
X1_=DAF(Z1_1,label_daf,10,"sigmoid")


X1=np.column_stack(())


