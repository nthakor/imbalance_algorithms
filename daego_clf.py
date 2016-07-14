import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score,roc_auc_score,recall_score
from algorithms.utils import _read_split,_class_split,_one_hot,_f_count,process_cm
from algorithms.daego import DAEGO
from algorithms.TFMLP import MLPR
from sklearn import tree
from sklearn.metrics import confusion_matrix

trX, teX, trY, teY = _read_split(
	"~/Google Drive/REU/classification-datasets/classification-datasets/boundary.csv",
	read=1,oneHot=0)

# #preprocessing 
# scaler=MinMaxScaler(feature_range=(0, 1))
# trX_scaled=scaler.fit_transform(trX)

X0,X1=_class_split(trX,trY,oneHot=0)

layer_daego=[X0.shape[1],200,250]
syn_X=DAEGO(X1,layer_daego,500,10)

X1=np.vstack((X1,syn_X))
X1=np.column_stack((X1,np.ones(X1.shape[0])))
X0=np.column_stack((X0,np.zeros(X0.shape[0])))
Xy=np.vstack((X0,X1))
np.random.shuffle(Xy)
trY=Xy[:,Xy.shape[1]-1]
trX=np.delete(Xy,Xy.shape[1]-1,axis=1)

"""

TRC0,TRC1=_class_split(trX,trY)
TEC0,TEC1=_class_split(teX,teY)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

trc0=pca.fit_transform(TRC0)
trc1=pca.fit_transform(TRC1)
tec0=pca.fit_transform(TEC0)
tec1=pca.fit_transform(TEC1)


plt.scatter(trc0[:,0],trc0[:,1],c='b',s=10,label="training_0")
plt.scatter(trc1[:,0],trc1[:,1],c='r',s=10,label="training_1")
plt.scatter(tec0[:,0],tec0[:,1],c='g',s=10,label="test_0")
plt.scatter(tec1[:,0],tec1[:,1],c='grey',s=10,label="test_1")
plt.legend()

"""


print _f_count(teY),"test f count"
clf = tree.DecisionTreeRegressor()
clf = clf.fit(trX, trY)
pred=clf.predict(teX)
pred=pred.astype(np.int32)
teY=teY.astype(np.int32)
print _f_count(pred),"pred f count"
conf_mat=confusion_matrix(teY, pred)

process_cm(conf_mat, to_print=True)
print precision_score(teY,pred),"Precision Score"
print recall_score(teY,pred),"Recall Score"
print roc_auc_score(teY,pred), "ROC_AUC"


# plt.show()