from sklearn import tree
from sklearn.metrics import confusion_matrix
from algorithms.utils import _read_split,_class_split,_one_hot,_f_count,process_cm
import numpy as np
from sklearn.metrics import precision_score,roc_auc_score,recall_score

trX, teX, trY, teY = _read_split(
	"~/Google Drive/REU/classification-datasets/classification-datasets/boundary.csv"
	,read=1,oneHot=0)


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