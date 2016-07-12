import numpy as np
from encoder import autoencoder
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


def DAEGO(X_s,H,P,batch_range):
	"""
Parameters
----------

X_s: small class features

H : layers (first layers shoud have same neurons as number of features)

P : percent oversampling

batch_range : size of minibatch


Returns
-------

syn_Z: synthetic sample with same number of features as smaller class
"""

	n_samples=int(X_s.shape[0]*P/100)
	print "generating %d samples" %(n_samples)
	x_init=np.random.randn(n_samples,X_s.shape[1])
	scaler=MinMaxScaler(feature_range=(-1,1))
	x_ini_scaled=scaler.fit_transform(x_init)
	ae=autoencoder(dimensions=H)
	learning_rate = 0.001
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])
	sess = tf.Session()
	sess.run(tf.initialize_all_variables())
	n_epoch=100
	for epoch_i in range(n_epoch):
	    for start, end in zip(range(0, len(X_s), batch_range),range(batch_range, len(X_s), batch_range)):
	        input_ = X_s[start:end]
	        sess.run(optimizer, feed_dict={ae['x']: input_, ae['corrupt_prob']: [1.0]})
	    # print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: X_s, ae['corrupt_prob']: [1.0]}))
	syn_Z = sess.run(ae['y'], feed_dict={ae['x']: x_ini_scaled, ae['corrupt_prob']: [0.0]})
	sess.close()
	return syn_Z

