import pylab as plt
from algorithms.utils import _read_dat,_class_split,_read_split
from algorithms.smote import SMOTE
from algorithms.daego import DAEGO
from visual.plot import plot_syn
from sklearn.decomposition import PCA
import warnings
# import matplotlib.pyplot as plt 

trX, teX, trY, teY = _read_split(
	"../datasets/nd-data/coil2000.csv",
	read=1,oneHot=0)

# from algorithms.utils import _read_dat
# trX, teX, trY, teY = _read_dat(
# 	"dataset/KDDCUP/DS/ds2.dat",skip=15,
# 	read=1,oneHot=0)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
trX=scaler.fit_transform(trX)
X0,X1=_class_split(trX,trY)

warnings.filterwarnings("ignore", 
	category=DeprecationWarning)
P=200
syn_S=SMOTE(X1, P, 5)


layer_daego=[100,150]
batch_range=50

inp_shape=[X1.shape[1]]
layer_daego=inp_shape+layer_daego
syn_D=DAEGO(X1,layer_daego,P,batch_range)

pca=PCA(n_components=2)
X=pca.fit_transform(X1)
sm=pca.fit_transform(syn_S)
dg=pca.fit_transform(syn_D)

plt.figure()
plt.subplot(211)
plt.title("SMOTE")
plt.scatter(X[:,0],X[:,1],c="#00ffff",label='(+)class')
plt.scatter(sm[:,0],sm[:,1],c="r",label='smote')
plt.legend()
plt.subplot(212)
plt.title("DAEGO")
plt.scatter(X[:,0],X[:,1],c="#00ffff",label='(+)class')
plt.scatter(dg[:,0],dg[:,1],c="g",label='daego')
plt.legend()
plt.savefig('sm_dg.png')
# plt.show()