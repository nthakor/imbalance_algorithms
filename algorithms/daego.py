import numpy as np
from encoder import autoencoder
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


def DAEGO(X_s,H,P,batch_range):
	n_samples=int(X_s.shape[0]*P/100)
	x_init=np.random.randn(n_sample,X_s.shape[1])
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
	    print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: X_s, ae['corrupt_prob']: [1.0]}))
	syn_Z = sess.run(ae['y'], feed_dict={ae['x']: x_ini_scaled, ae['corrupt_prob']: [0.0]})
	sess.close()
	return syn_Z

