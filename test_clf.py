import numpy as np
from deepautoencoder import StackedAutoEncoder
from algorithms.utils import _read_split,_class_split
from algorithms.clf_utils import _clf_dtree,_clf_svm,_clf_mlp


trX, teX, trY, teY = _read_split(
	"../datasets/nd-data/spls-lymphoma.csv",
	read=1,oneHot=0)

import numpy.linalg as LA
from sklearn.preprocessing import StandardScaler as StdScaler
from sklearn.preprocessing import normalize as norm
X0,X1=_class_split(trX,trY,oneHot=0)
scaler=StdScaler()
x_tr=scaler.fit_transform(X1.astype(float))
x_norm=norm(x_tr,axis=0)
P=100
n_samples=int(X1.shape[0]*P/100)
print "generating %d samples" %(n_samples)

norm_param=[LA.norm(x) for x in x_tr.T]
X_init=np.random.standard_normal(size=(n_samples,X1.shape[1]))
x_init_tr=scaler.transform(X_init)
x_ini_norm=norm(x_init_tr)

model = StackedAutoEncoder(dims=[200,250,200,X1.shape[1]], 
	activations=['sigmoid', 'sigmoid','sigmoid','sigmoid'],
	 noise='gaussian', epoch=[10000,10000,10000,10000],
     loss='rmse', lr=0.001, 
      batch_size=5, print_step=2000)
model.fit(x_norm)
x_init_encoded=model.transform(x_ini_norm)
x_init_norminv=np.multiply(x_init_encoded,norm_param)
syn_X=scaler.inverse_transform(x_init_norminv)
X1=np.vstack((X1,syn_X))
X1=np.column_stack((X1,np.ones(X1.shape[0])))
X0=np.column_stack((X0,np.zeros(X0.shape[0])))
Xy=np.vstack((X0,X1))
np.random.shuffle(Xy)
trY=Xy[:,Xy.shape[1]-1]
trX=np.delete(Xy,Xy.shape[1]-1,axis=1)


_clf_dtree(trX,teX,trY,teY)
# _clf_svm(trX,teX,trY,teY)
_clf_mlp(trX,teX,trY,teY)