import numpy as np
from sklearn import tree
from sklearn.metrics import precision_score,roc_auc_score,recall_score
from algorithms.utils import _read_split,_class_split,_one_hot,_f_count,process_cm

from algorithms.utils import _read_split
from algorithms.daf import DAF
from sklearn.metrics import confusion_matrix
trX, teX, trY, teY = _read_split("dataset/boundary.csv",read=1)


layer=[trX.shape[1],150,100,150,trX.shape[1]]

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


trX,trY=_daf_module(trX,trY,layer,10)
teX,teY=_daf_module(teX,teY,layer,10)


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
