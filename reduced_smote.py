import numpy as np
import warnings
from algorithms.utils import _read_split,_class_split,_read_dat,_merge_syn
from algorithms.clf_utils import _clf_dtree,_clf_svm,_clf_mlp
from algorithms.smote import SMOTE
from sklearn.preprocessing import MinMaxScaler as scaler
from deepautoencoder import StackedAutoEncoder

trX, teX, trY, teY = _read_dat("dataset/page-blocks0.dat",skip=15,read=1,oneHot=0)

d=int(trX.shape[1]/2)
model = StackedAutoEncoder(dims=[d], activations=['tanh'], noise='gaussian', epoch=[10000],
                            loss='cross-entropy', lr=0.007, batch_size=10, print_step=5000)
trX = model.fit_transform(trX)

print trX.shape,"trX after process"
print teX.shape,"teX"
c0,c1=_class_split(trX,trY)
warnings.filterwarnings("ignore",category=DeprecationWarning)
syn_S=SMOTE(c1,200,5)

trX,trY=_merge_syn(c0,c1,syn_S)
# d=d*2
model1 = StackedAutoEncoder(dims=[d*2], activations=['tanh'], noise='gaussian', epoch=[10000],
                            loss='cross-entropy', lr=0.007, batch_size=10, print_step=5000)
trX = model1.fit_transform(trX)

_clf_dtree(trX,teX,trY,teY)