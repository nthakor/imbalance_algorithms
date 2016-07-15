import numpy as np
from sklearn.metrics import precision_score,roc_auc_score,recall_score,confusion_matrix
from algorithms.utils import _read_split,_class_split,_one_hot,_f_count,process_cm


trX, teX, trY, teY = _read_split("dataset/boundary.csv",read=1)

from mlxtend.tf_classifier import TfMultiLayerPerceptron
mlp = TfMultiLayerPerceptron(eta=0.01, 
                             epochs=100, 
                             hidden_layers=[1000, 500],
                             activations=['relu', 'relu'],
                             print_progress=3, 
                             minibatches=15, 
                             optimizer='adam',
                             random_seed=1)



delIdx=8

while(delIdx):
	trX=np.delete(trX,-1,axis=0)
	trY=np.delete(trY,-1,axis=0)
	delIdx=delIdx-1

teY=teY.astype(np.int32)
trY=trY.astype(np.int32)
mlp.fit(trX,trY)

pred=mlp.predict(teX)



print _f_count(teY),"test f count"
pred=pred.astype(np.int32)
print _f_count(pred),"pred f count"
conf_mat=confusion_matrix(teY, pred)

process_cm(conf_mat, to_print=True)
print precision_score(teY,pred),"Precision Score"
print recall_score(teY,pred),"Recall Score"
print roc_auc_score(teY,pred), "ROC_AUC"


print "##############################################"
print "##########DECISION TREE#######################"
print "##############################################"


from sklearn import tree


trX, teX, trY, teY = _read_split("dataset/boundary.csv",read=1)

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
