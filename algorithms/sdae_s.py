import numpy as np
import numpy.linalg as LA
from sklearn.preprocessing import StandardScaler as StdScaler
from sklearn.preprocessing import normalize as norm
from algorithms.deepautoencoder.stacked_autoencoder import StackedAutoEncoder

def sdae_syn(X_s,h_layer,activation,batch_size,P,noise='gaussian',loss='rmse',lr=0.001):
	scaler=StdScaler()
	x_tr=scaler.fit_transform(X_s.astype(float))
	x_norm=norm(x_tr,axis=0)

	n_samples=int(X_s.shape[0]*P/100)
	print "generating %d samples" %(n_samples)

	norm_param=[LA.norm(x) for x in x_tr.T]
	X_init=np.random.standard_normal(size=(n_samples,X_s.shape[1]))
	x_init_tr=scaler.transform(X_init)
	x_ini_norm=norm(x_init_tr)
	SDAE=SDAE=StackedAutoEncoder(dims=h_layer,
		activations=activation,noise=noise,epoch=[100 for i in range(len(h_layer))],batch_size=batch_size,lr=lr)
	SDAE.fit(x_norm)
	x_init_encoded=SDAE.transform(x_ini_norm)
	x_init_norminv=np.multiply(x_init_encoded,norm_param)
	syn_Z=scaler.inverse_transform(x_init_norminv)
	return syn_Z


