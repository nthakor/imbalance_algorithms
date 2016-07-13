import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings 
from algorithms.daego import DAEGO
from algorithms.smote import SMOTE
from algorithms.utils import _read_split,_class_split
import time


trX, teX, trY, teY = _read_split("~/Google Drive/REU/classification-datasets/classification-datasets/coil2000.csv",read=1,oneHot=0)
X0,X1=_class_split(trX,trY,oneHot=0)
layer_daego=[X0.shape[1],100,150]
t=time.time()
syn_D=DAEGO(X1,layer_daego,100,10)
td=time.time()-t
print "time for daego",td
warnings.filterwarnings("ignore", category=DeprecationWarning)
syn_S=SMOTE(X1, 100, 2)
print "time for smote",time.time()-td

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

x_O=pca.fit_transform(X1)
x_D=pca.fit_transform(syn_D)
x_S=pca.fit_transform(syn_S)
x_0=pca.fit_transform(X0)


plt.scatter(x_O[:,0],x_O[:,1],c='b',s=10,label="original")
plt.scatter(x_D[:,0],x_D[:,1],c='r',s=10,label="daego")
plt.scatter(x_S[:,0],x_S[:,1],c='g',s=10,label="smote")
plt.scatter(x_0[:,0],x_0[:,1],c='grey',s=10,label="false class")
plt.legend()
plt.show()