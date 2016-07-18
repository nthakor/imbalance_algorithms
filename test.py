import numpy as np
import warnings
from algorithms.utils import _read_split,_class_split
from algorithms.clf_utils import _clf_dtree,_clf_svm,_clf_mlp
from algorithms.smote import SMOTE

trX, teX, trY, teY = _read_split(
	"../datasets/nd-data/boundary.csv",
	read=1,oneHot=0)


X0,X1=_class_split(trX,trY,oneHot=0)

from sklearn.metrics import silhouette_score

print silhouette_score(trX,trY),"training"
print silhouette_score(teX,teY),"test"